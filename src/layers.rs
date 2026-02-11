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

#[cfg(target_arch = "x86_64")]
unsafe fn hsum_256(x: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(x, 1);
    let lo = _mm256_castsi256_si128(x);
    let sum = _mm_add_epi32(lo, hi);
    let sum = _mm_hadd_epi32(sum, sum);
    let sum = _mm_hadd_epi32(sum, sum);
    _mm_cvtsi128_si32(sum)
}

pub struct AffineTransform {
    pub biases: Vec<i32>,
    pub weights: Vec<i8>,
    pub input_dims: usize,
    pub output_dims: usize,
    pub padded_input_dims: usize,
}

impl AffineTransform {
    pub fn new(input_dims: usize, output_dims: usize) -> Self {
        let padded_input_dims = (input_dims + 31) / 32 * 32;
        Self {
            biases: vec![0; output_dims],
            weights: vec![0; output_dims * padded_input_dims],
            input_dims,
            output_dims,
            padded_input_dims,
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn propagate_avx2(&self, input: &[u8], output: &mut [i32]) {
        let num_chunks = self.padded_input_dims / 32;

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
                    let input_vec = _mm256_loadu_si256(in_ptr as *const _);

                    let w0 = _mm256_loadu_si256(w_ptr0.add(c * 32) as *const _);
                    let w1 = _mm256_loadu_si256(w_ptr1.add(c * 32) as *const _);
                    let w2 = _mm256_loadu_si256(w_ptr2.add(c * 32) as *const _);
                    let w3 = _mm256_loadu_si256(w_ptr3.add(c * 32) as *const _);

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
                        let input_vec = _mm256_loadu_si256(in_ptr as *const _);
                        let w = _mm256_loadu_si256(w_ptr.add(c * 32) as *const _);
                        let p = _mm256_maddubs_epi16(input_vec, w);
                        let s = _mm256_madd_epi16(p, ones);
                        acc = _mm256_add_epi32(acc, s);
                    }
                    output[r_idx] = hsum_256(acc) + self.biases[r_idx];
                }
            }
        }
    }
}

impl Layer for AffineTransform {
    type Input = u8;
    type Output = i32;

    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return self.propagate_avx2(input, output);
            }
        }

        // Fallback implementation (row-major weights)
        output.copy_from_slice(&self.biases);

        for i in 0..self.input_dims {
            let in_val = input[i];
            if in_val == 0 {
                continue;
            }
            let in_val_i32 = in_val as i32;

            for j in 0..self.output_dims {
                // weights[j][i]
                let weight_idx = j * self.padded_input_dims + i;
                let w = self.weights[weight_idx] as i32;
                output[j] += w * in_val_i32;
            }
        }
    }

    fn read_parameters<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
        let biases = read_i32_array(reader, self.output_dims)?;
        self.biases = biases;
        let weights_raw = read_i8_array(reader, self.output_dims * self.padded_input_dims)?;
        self.weights = weights_raw;
        Ok(())
    }
}

pub type AffineTransformSparseInput = AffineTransform;

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
            let vec = _mm256_loadu_si256(input.as_ptr().add(i) as *const _);
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

    fn propagate(&self, input: &[i32], output: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return self.propagate_avx2(input, output);
            }
        }

        for (i, &val) in input.iter().enumerate().take(self.dims) {
            let scaled = val >> 6; // WeightScaleBits = 6
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
            let vec = _mm256_loadu_si256(input.as_ptr().add(i) as *const _);

            let even_sq = _mm256_mul_epi32(vec, vec);
            let even_res = _mm256_srai_epi64(even_sq, 19);

            let vec_odd = _mm256_shuffle_epi32(vec, 0xF5); // _MM_SHUFFLE(3, 3, 1, 1)
            let odd_sq = _mm256_mul_epi32(vec_odd, vec_odd);
            let odd_res = _mm256_srai_epi64(odd_sq, 19);

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

    fn propagate(&self, input: &[i32], output: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return self.propagate_avx2(input, output);
            }
        }

        for (i, &val) in input.iter().enumerate().take(self.dims) {
            let val_i64 = val as i64;
            let squared = val_i64 * val_i64;
            // >> (2 * 6 + 7) = 19
            let scaled = squared >> 19;
            output[i] = scaled.clamp(0, 127) as u8;
        }
    }

    fn read_parameters<R: Read>(&mut self, _reader: &mut R) -> io::Result<()> {
        Ok(())
    }
}
