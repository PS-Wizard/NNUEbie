use crate::aligned::AlignedBuffer;
use crate::loader::{read_i32_array, read_i8_array};
use std::io::{self, Read};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub trait Layer {
    type Input;
    type Output;
    fn propagate(&self, input: &[Self::Input], output: &mut [Self::Output]);
    fn read_parameters<R: Read>(&mut self, reader: &mut R) -> io::Result<()>;
}

#[cfg(all(target_arch = "x86_64", not(feature = "simd_avx512")))]
unsafe fn hsum_256(x: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(x, 1);
    let lo = _mm256_castsi256_si128(x);
    let sum = _mm_add_epi32(lo, hi);
    let sum = _mm_hadd_epi32(sum, sum);
    let sum = _mm_hadd_epi32(sum, sum);
    _mm_cvtsi128_si32(sum)
}

pub struct AffineTransform {
    pub biases: AlignedBuffer<i32>,
    pub weights: AlignedBuffer<i8>,
    pub input_dims: usize,
    pub output_dims: usize,
    pub padded_input_dims: usize,
}

impl AffineTransform {
    pub fn new(input_dims: usize, output_dims: usize) -> Self {
        let padded_input_dims = input_dims.div_ceil(32) * 32;
        Self {
            biases: AlignedBuffer::new(output_dims),
            weights: AlignedBuffer::new(output_dims * padded_input_dims),
            input_dims,
            output_dims,
            padded_input_dims,
        }
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "simd_avx512")))]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_avx2_32x32(&self, input: &[u8], output: &mut [i32]) {
        debug_assert_eq!(self.output_dims, 32);
        debug_assert_eq!(self.padded_input_dims, 32);

        let input_vec = _mm256_load_si256(input.as_ptr() as *const _);
        let ones = _mm256_set1_epi16(1);
        let w_ptr = self.weights.as_ptr();

        for r in (0..32).step_by(4) {
            let row = r * 32;

            let w0 = _mm256_load_si256(w_ptr.add(row) as *const _);
            let w1 = _mm256_load_si256(w_ptr.add(row + 32) as *const _);
            let w2 = _mm256_load_si256(w_ptr.add(row + 64) as *const _);
            let w3 = _mm256_load_si256(w_ptr.add(row + 96) as *const _);

            let p0 = _mm256_maddubs_epi16(input_vec, w0);
            let p1 = _mm256_maddubs_epi16(input_vec, w1);
            let p2 = _mm256_maddubs_epi16(input_vec, w2);
            let p3 = _mm256_maddubs_epi16(input_vec, w3);

            let s0 = _mm256_madd_epi16(p0, ones);
            let s1 = _mm256_madd_epi16(p1, ones);
            let s2 = _mm256_madd_epi16(p2, ones);
            let s3 = _mm256_madd_epi16(p3, ones);

            output[r] = hsum_256(s0) + self.biases[r];
            output[r + 1] = hsum_256(s1) + self.biases[r + 1];
            output[r + 2] = hsum_256(s2) + self.biases[r + 2];
            output[r + 3] = hsum_256(s3) + self.biases[r + 3];
        }
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "simd_avx512")))]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_avx2(&self, input: &[u8], output: &mut [i32]) {
        if self.output_dims == 32 && self.padded_input_dims == 32 {
            self.propagate_avx2_32x32(input, output);
            return;
        }

        let num_chunks = self.padded_input_dims / 32;

        if self.output_dims == 1 {
            let mut acc = _mm256_setzero_si256();
            let w_ptr = self.weights.as_ptr();
            let ones = _mm256_set1_epi16(1);

            for c in 0..num_chunks {
                let in_ptr = input.as_ptr().add(c * 32);
                let input_vec = _mm256_load_si256(in_ptr as *const _);
                let w = _mm256_load_si256(w_ptr.add(c * 32) as *const _);
                let p = _mm256_maddubs_epi16(input_vec, w);
                let s = _mm256_madd_epi16(p, ones);
                acc = _mm256_add_epi32(acc, s);
            }

            output[0] = hsum_256(acc) + self.biases[0];
            return;
        }

        for r in (0..self.output_dims).step_by(4) {
            let left = self.output_dims - r;
            if left >= 4 {
                let mut acc0 = _mm256_setzero_si256();
                let mut acc1 = _mm256_setzero_si256();
                let mut acc2 = _mm256_setzero_si256();
                let mut acc3 = _mm256_setzero_si256();

                let w_ptr0 = self.weights.as_ptr().add(r * self.padded_input_dims);
                let w_ptr1 = w_ptr0.add(self.padded_input_dims);
                let w_ptr2 = w_ptr1.add(self.padded_input_dims);
                let w_ptr3 = w_ptr2.add(self.padded_input_dims);
                let ones = _mm256_set1_epi16(1);

                for c in 0..num_chunks {
                    let in_ptr = input.as_ptr().add(c * 32);
                    let input_vec = _mm256_load_si256(in_ptr as *const _);

                    let w0 = _mm256_load_si256(w_ptr0.add(c * 32) as *const _);
                    let w1 = _mm256_load_si256(w_ptr1.add(c * 32) as *const _);
                    let w2 = _mm256_load_si256(w_ptr2.add(c * 32) as *const _);
                    let w3 = _mm256_load_si256(w_ptr3.add(c * 32) as *const _);

                    // maddubs: input unsigned, weight signed
                    let p0 = _mm256_maddubs_epi16(input_vec, w0);
                    let p1 = _mm256_maddubs_epi16(input_vec, w1);
                    let p2 = _mm256_maddubs_epi16(input_vec, w2);
                    let p3 = _mm256_maddubs_epi16(input_vec, w3);

                    let s0 = _mm256_madd_epi16(p0, ones);
                    let s1 = _mm256_madd_epi16(p1, ones);
                    let s2 = _mm256_madd_epi16(p2, ones);
                    let s3 = _mm256_madd_epi16(p3, ones);

                    acc0 = _mm256_add_epi32(acc0, s0);
                    acc1 = _mm256_add_epi32(acc1, s1);
                    acc2 = _mm256_add_epi32(acc2, s2);
                    acc3 = _mm256_add_epi32(acc3, s3);
                }

                output[r] = hsum_256(acc0) + self.biases[r];
                output[r + 1] = hsum_256(acc1) + self.biases[r + 1];
                output[r + 2] = hsum_256(acc2) + self.biases[r + 2];
                output[r + 3] = hsum_256(acc3) + self.biases[r + 3];
            } else {
                for k in 0..left {
                    let r_idx = r + k;
                    let mut acc = _mm256_setzero_si256();
                    let w_ptr = self.weights.as_ptr().add(r_idx * self.padded_input_dims);
                    let ones = _mm256_set1_epi16(1);

                    for c in 0..num_chunks {
                        let in_ptr = input.as_ptr().add(c * 32);
                        let input_vec = _mm256_load_si256(in_ptr as *const _);

                        let w = _mm256_load_si256(w_ptr.add(c * 32) as *const _);
                        let p = _mm256_maddubs_epi16(input_vec, w);
                        let s = _mm256_madd_epi16(p, ones);
                        acc = _mm256_add_epi32(acc, s);
                    }
                    output[r_idx] = hsum_256(acc) + self.biases[r_idx];
                }
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "simd_avx512",
        not(feature = "simd_avx512_vnni")
    ))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn add_dpbusd_512(acc: __m512i, a: __m512i, b: __m512i) -> __m512i {
        let product = _mm512_maddubs_epi16(a, b);
        let summed = _mm512_madd_epi16(product, _mm512_set1_epi16(1));
        _mm512_add_epi32(acc, summed)
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd_avx512_vnni"))]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn add_dpbusd_512_vnni(acc: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpbusd_epi32(acc, a, b)
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "simd_avx512",
        not(feature = "simd_avx512_vnni")
    ))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn propagate_avx512(&self, input: &[u8], output: &mut [i32]) {
        if self.output_dims == 1 {
            let in256 = _mm256_load_si256(input.as_ptr() as *const _);
            let w256 = _mm256_load_si256(self.weights.as_ptr() as *const _);
            let in512 = _mm512_zextsi256_si512(in256);
            let w512 = _mm512_zextsi256_si512(w256);
            let acc = _mm512_setzero_si512();
            let sum = Self::add_dpbusd_512(acc, in512, w512);
            output[0] = _mm512_reduce_add_epi32(sum) + self.biases[0];
            return;
        }

        debug_assert_eq!(self.output_dims, 32);
        let num_chunks = self.padded_input_dims / 4;
        let input32 = input.as_ptr() as *const i32;

        let bias_ptr = self.biases.as_ptr() as *const __m512i;
        let mut acc0 = _mm512_load_si512(bias_ptr);
        let mut acc1 = _mm512_load_si512(bias_ptr.add(1));

        let weights_ptr = self.weights.as_ptr();
        let block_stride = self.output_dims * 4;

        for i in 0..num_chunks {
            let in_val = *input32.add(i);

            let in_vec = _mm512_set1_epi32(in_val);
            let col_ptr = weights_ptr.add(i * block_stride);
            let w0 = _mm512_load_si512(col_ptr as *const _);
            let w1 = _mm512_load_si512(col_ptr.add(64) as *const _);

            acc0 = Self::add_dpbusd_512(acc0, in_vec, w0);
            acc1 = Self::add_dpbusd_512(acc1, in_vec, w1);
        }

        let out_ptr = output.as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(out_ptr, acc0);
        _mm512_store_si512(out_ptr.add(1), acc1);
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd_avx512_vnni"))]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn propagate_avx512_vnni(&self, input: &[u8], output: &mut [i32]) {
        if self.output_dims == 1 {
            let in256 = _mm256_load_si256(input.as_ptr() as *const _);
            let w256 = _mm256_load_si256(self.weights.as_ptr() as *const _);
            let in512 = _mm512_zextsi256_si512(in256);
            let w512 = _mm512_zextsi256_si512(w256);
            let acc = _mm512_setzero_si512();
            let sum = Self::add_dpbusd_512_vnni(acc, in512, w512);
            output[0] = _mm512_reduce_add_epi32(sum) + self.biases[0];
            return;
        }

        debug_assert_eq!(self.output_dims, 32);
        let num_chunks = self.padded_input_dims / 4;
        let input32 = input.as_ptr() as *const i32;

        let bias_ptr = self.biases.as_ptr() as *const __m512i;
        let mut acc0 = _mm512_load_si512(bias_ptr);
        let mut acc1 = _mm512_load_si512(bias_ptr.add(1));

        let weights_ptr = self.weights.as_ptr();
        let block_stride = self.output_dims * 4;

        for i in 0..num_chunks {
            let in_val = *input32.add(i);

            let in_vec = _mm512_set1_epi32(in_val);
            let col_ptr = weights_ptr.add(i * block_stride);
            let w0 = _mm512_load_si512(col_ptr as *const _);
            let w1 = _mm512_load_si512(col_ptr.add(64) as *const _);

            acc0 = Self::add_dpbusd_512_vnni(acc0, in_vec, w0);
            acc1 = Self::add_dpbusd_512_vnni(acc1, in_vec, w1);
        }

        let out_ptr = output.as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(out_ptr, acc0);
        _mm512_store_si512(out_ptr.add(1), acc1);
    }
}

impl Layer for AffineTransform {
    type Input = u8;
    type Output = i32;

    #[cfg(all(target_arch = "x86_64", feature = "simd_avx512_vnni"))]
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        unsafe {
            self.propagate_avx512_vnni(input, output);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "simd_avx512",
        not(feature = "simd_avx512_vnni")
    ))]
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        unsafe {
            self.propagate_avx512(input, output);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "simd_avx2",
        not(feature = "simd_avx512")
    ))]
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        unsafe {
            self.propagate_avx2(input, output);
        }
    }

    #[cfg(any(
        not(target_arch = "x86_64"),
        not(any(
            feature = "simd_avx2",
            feature = "simd_avx512",
            feature = "simd_avx512_vnni"
        ))
    ))]
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        output.copy_from_slice(&self.biases);

        for (i, &in_val) in input.iter().enumerate().take(self.input_dims) {
            if in_val == 0 {
                continue;
            }
            let in_val_i32 = in_val as i32;

            for (j, out_val) in output.iter_mut().enumerate().take(self.output_dims) {
                let weight_idx = j * self.padded_input_dims + i;
                let w = self.weights[weight_idx] as i32;
                *out_val += w * in_val_i32;
            }
        }
    }

    fn read_parameters<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
        let biases_vec = read_i32_array(reader, self.output_dims)?;
        self.biases = AlignedBuffer::from_vec(biases_vec);

        let weights_raw = read_i8_array(reader, self.output_dims * self.padded_input_dims)?;
        #[cfg(feature = "simd_avx512")]
        {
            let mut scrambled = vec![0i8; weights_raw.len()];
            for (i, &weight) in weights_raw.iter().enumerate() {
                let idx = get_weight_index_scrambled(i, self.padded_input_dims, self.output_dims);
                scrambled[idx] = weight;
            }
            self.weights = AlignedBuffer::from_vec(scrambled);
        }

        #[cfg(not(feature = "simd_avx512"))]
        {
            self.weights = AlignedBuffer::from_vec(weights_raw);
        }
        Ok(())
    }
}

fn get_fc_permutation_map(dims: usize) -> Vec<usize> {
    let mut map = vec![0; dims];
    for (i, m) in map.iter_mut().enumerate().take(dims) {
        let c = i / 32;
        let byte = i % 32;
        let k = c / 2;
        let r = c % 2;

        let (block_a, block_b) = if r == 0 {
            (4 * k, 4 * k + 2)
        } else {
            (4 * k + 1, 4 * k + 3)
        };

        let feature_idx = if byte < 8 {
            block_a * 16 + byte
        } else if byte < 16 {
            block_b * 16 + (byte - 8)
        } else if byte < 24 {
            block_a * 16 + (byte - 16) + 8
        } else {
            block_b * 16 + (byte - 24) + 8
        };
        *m = feature_idx;
    }
    map
}

fn permute_fc_weights_data(
    input_dims: usize,
    padded_input_dims: usize,
    output_dims: usize,
    weights: &[i8],
) -> Vec<i8> {
    let map = get_fc_permutation_map(input_dims);
    let mut permuted = vec![0i8; output_dims * padded_input_dims];

    for r in 0..output_dims {
        let row_offset = r * padded_input_dims;
        for c in 0..input_dims {
            permuted[row_offset + c] = weights[row_offset + map[c]];
        }
        permuted[(row_offset + input_dims)..(row_offset + padded_input_dims)]
            .copy_from_slice(&weights[(row_offset + input_dims)..(row_offset + padded_input_dims)]);
    }

    permuted
}

fn get_weight_index_scrambled(i: usize, padded_input_dims: usize, output_dims: usize) -> usize {
    let chunk_size = 4;
    (i / chunk_size) % (padded_input_dims / chunk_size) * output_dims * chunk_size
        + i / padded_input_dims * chunk_size
        + i % chunk_size
}

pub struct AffineTransformSparseInput {
    pub biases: AlignedBuffer<i32>,
    pub weights: AlignedBuffer<i8>,
    pub input_dims: usize,
    pub output_dims: usize,
    pub padded_input_dims: usize,
}

impl AffineTransformSparseInput {
    pub fn new(input_dims: usize, output_dims: usize) -> Self {
        let padded_input_dims = input_dims.div_ceil(32) * 32;
        Self {
            biases: AlignedBuffer::new(output_dims),
            weights: AlignedBuffer::new(output_dims * padded_input_dims),
            input_dims,
            output_dims,
            padded_input_dims,
        }
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "simd_avx512")))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_dpbusd(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        let product = _mm256_maddubs_epi16(a, b);
        let summed = _mm256_madd_epi16(product, _mm256_set1_epi16(1));
        _mm256_add_epi32(acc, summed)
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "simd_avx512",
        not(feature = "simd_avx512_vnni")
    ))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn add_dpbusd_512(acc: __m512i, a: __m512i, b: __m512i) -> __m512i {
        let product = _mm512_maddubs_epi16(a, b);
        let summed = _mm512_madd_epi16(product, _mm512_set1_epi16(1));
        _mm512_add_epi32(acc, summed)
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd_avx512_vnni"))]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn add_dpbusd_512_vnni(acc: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpbusd_epi32(acc, a, b)
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "simd_avx512",
        not(feature = "simd_avx512_vnni")
    ))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn propagate_avx512(&self, input: &[u8], output: &mut [i32]) {
        debug_assert_eq!(input.len(), self.input_dims);
        debug_assert_eq!(output.len(), self.output_dims);
        debug_assert_eq!(self.output_dims, 16);

        let num_chunks = self.padded_input_dims / 4;
        let input32 = input.as_ptr() as *const i32;

        let acc = _mm512_load_si512(self.biases.as_ptr() as *const _);

        let mut sum = acc;
        let weights_ptr = self.weights.as_ptr();
        let block_stride = self.output_dims * 4;

        for i in 0..num_chunks {
            let in_val = *input32.add(i);
            if in_val == 0 {
                continue;
            }

            let in_vec = _mm512_set1_epi32(in_val);
            let col_ptr = weights_ptr.add(i * block_stride);
            let w = _mm512_load_si512(col_ptr as *const _);

            sum = Self::add_dpbusd_512(sum, in_vec, w);
        }

        _mm512_store_si512(output.as_mut_ptr() as *mut _, sum);
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd_avx512_vnni"))]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn propagate_avx512_vnni(&self, input: &[u8], output: &mut [i32]) {
        debug_assert_eq!(input.len(), self.input_dims);
        debug_assert_eq!(output.len(), self.output_dims);
        debug_assert_eq!(self.output_dims, 16);

        let num_chunks = self.padded_input_dims / 4;
        let input32 = input.as_ptr() as *const i32;

        let acc = _mm512_load_si512(self.biases.as_ptr() as *const _);

        let mut sum = acc;
        let weights_ptr = self.weights.as_ptr();
        let block_stride = self.output_dims * 4;

        for i in 0..num_chunks {
            let in_val = *input32.add(i);
            if in_val == 0 {
                continue;
            }

            let in_vec = _mm512_set1_epi32(in_val);
            let col_ptr = weights_ptr.add(i * block_stride);
            let w = _mm512_load_si512(col_ptr as *const _);

            sum = Self::add_dpbusd_512_vnni(sum, in_vec, w);
        }

        _mm512_store_si512(output.as_mut_ptr() as *mut _, sum);
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "simd_avx512")))]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_avx2(&self, input: &[u8], output: &mut [i32]) {
        debug_assert_eq!(input.len(), self.input_dims);
        debug_assert_eq!(output.len(), self.output_dims);
        debug_assert_eq!(self.output_dims % 8, 0);

        let num_chunks = self.padded_input_dims / 4;
        let input32 = input.as_ptr() as *const i32;

        let bias_ptr = self.biases.as_ptr() as *const __m256i;
        let mut acc0 = _mm256_load_si256(bias_ptr);
        let mut acc1 = _mm256_load_si256(bias_ptr.add(1));

        let weights_ptr = self.weights.as_ptr();
        let block_stride = self.output_dims * 4;

        for i in 0..num_chunks {
            let in_val = *input32.add(i);
            if in_val == 0 {
                continue;
            }

            let in_vec = _mm256_set1_epi32(in_val);
            let col_ptr = weights_ptr.add(i * block_stride);

            let w0 = _mm256_load_si256(col_ptr as *const __m256i);
            let w1 = _mm256_load_si256(col_ptr.add(32) as *const __m256i);

            acc0 = Self::add_dpbusd(acc0, in_vec, w0);
            acc1 = Self::add_dpbusd(acc1, in_vec, w1);
        }

        let out_ptr = output.as_mut_ptr() as *mut __m256i;
        _mm256_store_si256(out_ptr, acc0);
        _mm256_store_si256(out_ptr.add(1), acc1);
    }
}

impl Layer for AffineTransformSparseInput {
    type Input = u8;
    type Output = i32;

    #[cfg(all(target_arch = "x86_64", feature = "simd_avx512_vnni"))]
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        unsafe {
            self.propagate_avx512_vnni(input, output);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "simd_avx512",
        not(feature = "simd_avx512_vnni")
    ))]
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        unsafe {
            self.propagate_avx512(input, output);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "simd_avx2",
        not(feature = "simd_avx512")
    ))]
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        unsafe {
            self.propagate_avx2(input, output);
        }
    }

    #[cfg(any(
        not(target_arch = "x86_64"),
        not(any(
            feature = "simd_avx2",
            feature = "simd_avx512",
            feature = "simd_avx512_vnni"
        ))
    ))]
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        output.copy_from_slice(&self.biases);
        for (i, &in_val) in input.iter().enumerate().take(self.input_dims) {
            if in_val == 0 {
                continue;
            }
            let in_val_i32 = in_val as i32;
            for (j, out_val) in output.iter_mut().enumerate().take(self.output_dims) {
                let weight_idx = get_weight_index_scrambled(
                    j * self.padded_input_dims + i,
                    self.padded_input_dims,
                    self.output_dims,
                );
                let w = self.weights[weight_idx] as i32;
                *out_val += w * in_val_i32;
            }
        }
    }

    fn read_parameters<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
        let biases_vec = read_i32_array(reader, self.output_dims)?;
        self.biases = AlignedBuffer::from_vec(biases_vec);

        let weights_raw = read_i8_array(reader, self.output_dims * self.padded_input_dims)?;
        let permuted = permute_fc_weights_data(
            self.input_dims,
            self.padded_input_dims,
            self.output_dims,
            &weights_raw,
        );

        let mut scrambled = vec![0i8; self.output_dims * self.padded_input_dims];
        for (i, &weight) in permuted
            .iter()
            .enumerate()
            .take(self.output_dims * self.padded_input_dims)
        {
            let idx = get_weight_index_scrambled(i, self.padded_input_dims, self.output_dims);
            scrambled[idx] = weight;
        }

        self.weights = AlignedBuffer::from_vec(scrambled);
        Ok(())
    }
}

pub struct ClippedReLU {
    pub dims: usize,
}

impl ClippedReLU {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_avx2(&self, input: &[i32], output: &mut [u8]) {
        let n = self.dims / 8 * 8;
        for i in (0..n).step_by(8) {
            let vec = _mm256_load_si256(input.as_ptr().add(i) as *const _);
            let scaled = _mm256_srai_epi32(vec, 6);

            let lo = _mm256_castsi256_si128(scaled);
            let hi = _mm256_extracti128_si256(scaled, 1);
            let p = _mm_packus_epi32(lo, hi);
            let p2 = _mm_packus_epi16(p, p);

            let clamped = _mm_min_epu8(p2, _mm_set1_epi8(127));
            let val = _mm_cvtsi128_si64(clamped);
            *(output.as_mut_ptr().add(i) as *mut i64) = val;
        }

        for i in n..self.dims {
            let val = input[i];
            let scaled = val >> 6;
            output[i] = scaled.clamp(0, 127) as u8;
        }
    }
}

impl Layer for ClippedReLU {
    type Input = i32;
    type Output = u8;

    #[cfg(all(
        target_arch = "x86_64",
        any(
            feature = "simd_avx2",
            feature = "simd_avx512",
            feature = "simd_avx512_vnni"
        )
    ))]
    fn propagate(&self, input: &[i32], output: &mut [u8]) {
        unsafe {
            self.propagate_avx2(input, output);
        }
    }

    #[cfg(any(
        not(target_arch = "x86_64"),
        not(any(
            feature = "simd_avx2",
            feature = "simd_avx512",
            feature = "simd_avx512_vnni"
        ))
    ))]
    fn propagate(&self, input: &[i32], output: &mut [u8]) {
        for (i, &val) in input.iter().enumerate().take(self.dims) {
            let scaled = val >> 6;
            output[i] = scaled.clamp(0, 127) as u8;
        }
    }

    fn read_parameters<R: Read>(&mut self, _reader: &mut R) -> io::Result<()> {
        Ok(())
    }
}

pub struct SqrClippedReLU {
    pub dims: usize,
}

impl SqrClippedReLU {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_avx2(&self, input: &[i32], output: &mut [u8]) {
        let n = self.dims / 8 * 8;
        for i in (0..n).step_by(8) {
            let vec = _mm256_load_si256(input.as_ptr().add(i) as *const _);

            let even_sq = _mm256_mul_epi32(vec, vec);
            let even_res = _mm256_srli_epi64(even_sq, 19);

            let vec_odd = _mm256_shuffle_epi32(vec, 0xF5); // _MM_SHUFFLE(3, 3, 1, 1)
            let odd_sq = _mm256_mul_epi32(vec_odd, vec_odd);
            let odd_res = _mm256_srli_epi64(odd_sq, 19);

            let e_shuf = _mm256_shuffle_epi32(even_res, 0xD8); // _MM_SHUFFLE(3, 1, 2, 0)
            let o_shuf = _mm256_shuffle_epi32(odd_res, 0xD8);

            let unpacked = _mm256_unpacklo_epi32(e_shuf, o_shuf);

            let lo = _mm256_castsi256_si128(unpacked);
            let hi = _mm256_extracti128_si256(unpacked, 1);
            let p = _mm_packus_epi32(lo, hi);
            let p2 = _mm_packus_epi16(p, p);

            let clamped = _mm_min_epu8(p2, _mm_set1_epi8(127));
            let val = _mm_cvtsi128_si64(clamped);
            *(output.as_mut_ptr().add(i) as *mut i64) = val;
        }

        for i in n..self.dims {
            let val = input[i];
            let val_i64 = val as i64;
            let squared = val_i64 * val_i64;
            let scaled = squared >> 19;
            output[i] = scaled.clamp(0, 127) as u8;
        }
    }
}

impl Layer for SqrClippedReLU {
    type Input = i32;
    type Output = u8;

    #[cfg(all(
        target_arch = "x86_64",
        any(
            feature = "simd_avx2",
            feature = "simd_avx512",
            feature = "simd_avx512_vnni"
        )
    ))]
    fn propagate(&self, input: &[i32], output: &mut [u8]) {
        unsafe {
            self.propagate_avx2(input, output);
        }
    }

    #[cfg(any(
        not(target_arch = "x86_64"),
        not(any(
            feature = "simd_avx2",
            feature = "simd_avx512",
            feature = "simd_avx512_vnni"
        ))
    ))]
    fn propagate(&self, input: &[i32], output: &mut [u8]) {
        for (i, &val) in input.iter().enumerate().take(self.dims) {
            let val_i64 = val as i64;
            let squared = val_i64 * val_i64;
            let scaled = squared >> 19;
            output[i] = scaled.clamp(0, 127) as u8;
        }
    }

    fn read_parameters<R: Read>(&mut self, _reader: &mut R) -> io::Result<()> {
        Ok(())
    }
}
