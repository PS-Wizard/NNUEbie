use crate::aligned::AlignedBuffer;
use crate::feature_transformer::FeatureTransformer;
use crate::features::{self, make_index};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

type FeatureUpdateFn = unsafe fn(&mut [i16], &[i16]);

#[derive(Clone)]
pub struct Accumulator<const SIZE: usize> {
    // Use heap-allocated aligned memory
    pub accumulation: [AlignedBuffer<i16>; 2],
    pub psqt_accumulation: [[i32; 8]; 2],
    computed: [bool; 2],
    add_feature_fn: FeatureUpdateFn,
    remove_feature_fn: FeatureUpdateFn,
}

impl<const SIZE: usize> Default for Accumulator<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SIZE: usize> Accumulator<SIZE> {
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        let (add_fn, remove_fn) = if is_x86_feature_detected!("avx2") {
            (
                add_feature_avx2 as FeatureUpdateFn,
                remove_feature_avx2 as FeatureUpdateFn,
            )
        } else {
            (
                add_feature_scalar as FeatureUpdateFn,
                remove_feature_scalar as FeatureUpdateFn,
            )
        };

        #[cfg(not(target_arch = "x86_64"))]
        let (add_fn, remove_fn) = (
            add_feature_scalar as FeatureUpdateFn,
            remove_feature_scalar as FeatureUpdateFn,
        );

        // Allocate aligned memory on heap
        let acc0 = AlignedBuffer::<i16>::new(SIZE);
        let acc1 = AlignedBuffer::<i16>::new(SIZE);

        Self {
            accumulation: [acc0, acc1],
            psqt_accumulation: [[0; 8]; 2],
            computed: [false, false],
            add_feature_fn: add_fn,
            remove_feature_fn: remove_fn,
        }
    }

    /// Get reference to accumulator for a perspective
    pub fn get(&self, perspective: usize) -> &[i16] {
        &*self.accumulation[perspective]
    }

    /// Get mutable reference to accumulator for a perspective
    pub fn get_mut(&mut self, perspective: usize) -> &mut [i16] {
        &mut *self.accumulation[perspective]
    }

    // Refresh accumulator from scratch
    // pos_pieces: Iterator of (Square, Piece)
    // ksq: [Square; 2] (White King, Black King)
    pub fn refresh(
        &mut self,
        pieces: &[(usize, usize)], // (Square, Piece)
        ksq: [usize; 2],
        ft: &FeatureTransformer,
    ) {
        debug_assert_eq!(
            ft.half_dims, SIZE,
            "FeatureTransformer dims mismatch Accumulator size"
        );

        // Reset - copy biases to each perspective
        for c in 0..2 {
            self.accumulation[c].copy_from_slice(&ft.biases);
            self.psqt_accumulation[c].fill(0);
            self.computed[c] = true;
        }

        // Add features
        for &(sq, pc) in pieces {
            // For White Perspective
            let idx_w = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
            self.add_feature(features::WHITE, idx_w, ft);

            // For Black Perspective
            let idx_b = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
            self.add_feature(features::BLACK, idx_b, ft);
        }
    }

    /// Incrementally updates the accumulator.
    ///
    /// This function assumes that the King Squares (`ksq`) have NOT changed for the perspectives being updated.
    /// If a King has moved for a given color, you must use `refresh` for that color's accumulator instead,
    /// because the bucket and orientation for all pieces depend on the King's position.
    pub fn update_with_ksq(
        &mut self,
        added: &[(usize, usize)],   // (Square, Piece)
        removed: &[(usize, usize)], // (Square, Piece)
        ksq: [usize; 2],
        ft: &FeatureTransformer,
    ) {
        debug_assert_eq!(
            ft.half_dims, SIZE,
            "FeatureTransformer dims mismatch Accumulator size"
        );

        for &(sq, pc) in removed {
            // White Perspective
            let idx_w = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
            self.remove_feature(features::WHITE, idx_w, ft);

            // Black Perspective
            let idx_b = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
            self.remove_feature(features::BLACK, idx_b, ft);
        }

        for &(sq, pc) in added {
            // White Perspective
            let idx_w = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
            self.add_feature(features::WHITE, idx_w, ft);

            // Black Perspective
            let idx_b = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
            self.add_feature(features::BLACK, idx_b, ft);
        }
    }

    pub fn add_feature(&mut self, perspective: usize, feature_idx: usize, ft: &FeatureTransformer) {
        let half_dims = ft.half_dims;
        let offset = feature_idx * half_dims;
        let w_slice = &ft.weights[offset..offset + half_dims];

        // Use selected implementation
        unsafe {
            (self.add_feature_fn)(self.accumulation[perspective].as_mut_slice(), w_slice);
        }

        // PSQT update using SIMD
        self.update_psqt(perspective, feature_idx, ft, true);
    }

    /// Update PSQT accumulation using SIMD when available
    fn update_psqt(
        &mut self,
        perspective: usize,
        feature_idx: usize,
        ft: &FeatureTransformer,
        add: bool,
    ) {
        let psqt_offset = feature_idx * crate::feature_transformer::PSQT_BUCKETS;
        let psqt_slice =
            &ft.psqt_weights[psqt_offset..psqt_offset + crate::feature_transformer::PSQT_BUCKETS];

        // Compile-time AVX2 path
        #[cfg(all(target_arch = "x86_64", feature = "simd_avx2"))]
        unsafe {
            self.update_psqt_avx2(perspective, psqt_slice, add);
            return;
        }

        // Runtime detection path (when no compile-time feature set)
        #[cfg(all(
            target_arch = "x86_64",
            not(feature = "simd_avx2"),
            not(feature = "simd_scalar")
        ))]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.update_psqt_avx2(perspective, psqt_slice, add);
                return;
            }
        }

        // Scalar fallback
        if add {
            for (i, &pw) in psqt_slice.iter().enumerate() {
                self.psqt_accumulation[perspective][i] += pw;
            }
        } else {
            for (i, &pw) in psqt_slice.iter().enumerate() {
                self.psqt_accumulation[perspective][i] -= pw;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn update_psqt_avx2(&mut self, perspective: usize, psqt_slice: &[i32], add: bool) {
        let acc_ptr = self.psqt_accumulation[perspective].as_mut_ptr();
        let w_ptr = psqt_slice.as_ptr();

        // Load 8 i32 elements (256 bits) - covers all PSQT_BUCKETS
        let w = _mm256_loadu_si256(w_ptr as *const __m256i);
        let a = _mm256_loadu_si256(acc_ptr as *const __m256i);

        let res = if add {
            _mm256_add_epi32(a, w)
        } else {
            _mm256_sub_epi32(a, w)
        };

        _mm256_storeu_si256(acc_ptr as *mut __m256i, res);
    }

    fn remove_feature(&mut self, perspective: usize, feature_idx: usize, ft: &FeatureTransformer) {
        let half_dims = ft.half_dims;
        let offset = feature_idx * half_dims;
        let w_slice = &ft.weights[offset..offset + half_dims];

        // Use selected implementation
        unsafe {
            (self.remove_feature_fn)(self.accumulation[perspective].as_mut_slice(), w_slice);
        }

        // PSQT update using SIMD
        self.update_psqt(perspective, feature_idx, ft, false);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_feature_avx2(acc: &mut [i16], weights: &[i16]) {
    let mut i = 0;
    let acc_ptr = acc.as_mut_ptr();
    let w_ptr = weights.as_ptr();
    let count = acc.len();

    // Unroll by 4 (64 elements per iteration)
    // acc is AlignedBuffer, so aligned.
    // weights is from FeatureTransformer. If we align FT weights and stride is multiple of 32 (64 bytes), we can use aligned.
    // half_dims is 3072 or 128. 3072 * 2 = 6144 bytes (multiple of 64). 128 * 2 = 256 bytes (multiple of 64).
    // So weights slice start is aligned if weights buffer is aligned.

    // We will ensure FeatureTransformer uses AlignedBuffer.

    while i + 64 <= count {
        let w0 = _mm256_load_si256(w_ptr.add(i) as *const _);
        let w1 = _mm256_load_si256(w_ptr.add(i + 16) as *const _);
        let w2 = _mm256_load_si256(w_ptr.add(i + 32) as *const _);
        let w3 = _mm256_load_si256(w_ptr.add(i + 48) as *const _);

        let a0 = _mm256_load_si256(acc_ptr.add(i) as *const _);
        let a1 = _mm256_load_si256(acc_ptr.add(i + 16) as *const _);
        let a2 = _mm256_load_si256(acc_ptr.add(i + 32) as *const _);
        let a3 = _mm256_load_si256(acc_ptr.add(i + 48) as *const _);

        let r0 = _mm256_add_epi16(a0, w0);
        let r1 = _mm256_add_epi16(a1, w1);
        let r2 = _mm256_add_epi16(a2, w2);
        let r3 = _mm256_add_epi16(a3, w3);

        _mm256_store_si256(acc_ptr.add(i) as *mut _, r0);
        _mm256_store_si256(acc_ptr.add(i + 16) as *mut _, r1);
        _mm256_store_si256(acc_ptr.add(i + 32) as *mut _, r2);
        _mm256_store_si256(acc_ptr.add(i + 48) as *mut _, r3);

        i += 64;
    }

    // Remainder loop (if size not multiple of 64)
    while i + 16 <= count {
        let w = _mm256_load_si256(w_ptr.add(i) as *const _);
        let a = _mm256_load_si256(acc_ptr.add(i) as *const _);
        let res = _mm256_add_epi16(a, w);
        _mm256_store_si256(acc_ptr.add(i) as *mut _, res);
        i += 16;
    }

    // Scalar remainder
    for j in i..count {
        *acc_ptr.add(j) += *w_ptr.add(j) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn remove_feature_avx2(acc: &mut [i16], weights: &[i16]) {
    let mut i = 0;
    let acc_ptr = acc.as_mut_ptr();
    let w_ptr = weights.as_ptr();
    let count = acc.len();

    while i + 64 <= count {
        let w0 = _mm256_load_si256(w_ptr.add(i) as *const _);
        let w1 = _mm256_load_si256(w_ptr.add(i + 16) as *const _);
        let w2 = _mm256_load_si256(w_ptr.add(i + 32) as *const _);
        let w3 = _mm256_load_si256(w_ptr.add(i + 48) as *const _);

        let a0 = _mm256_load_si256(acc_ptr.add(i) as *const _);
        let a1 = _mm256_load_si256(acc_ptr.add(i + 16) as *const _);
        let a2 = _mm256_load_si256(acc_ptr.add(i + 32) as *const _);
        let a3 = _mm256_load_si256(acc_ptr.add(i + 48) as *const _);

        let r0 = _mm256_sub_epi16(a0, w0);
        let r1 = _mm256_sub_epi16(a1, w1);
        let r2 = _mm256_sub_epi16(a2, w2);
        let r3 = _mm256_sub_epi16(a3, w3);

        _mm256_store_si256(acc_ptr.add(i) as *mut _, r0);
        _mm256_store_si256(acc_ptr.add(i + 16) as *mut _, r1);
        _mm256_store_si256(acc_ptr.add(i + 32) as *mut _, r2);
        _mm256_store_si256(acc_ptr.add(i + 48) as *mut _, r3);

        i += 64;
    }

    while i + 16 <= count {
        let w = _mm256_load_si256(w_ptr.add(i) as *const _);
        let a = _mm256_load_si256(acc_ptr.add(i) as *const _);
        let res = _mm256_sub_epi16(a, w);
        _mm256_store_si256(acc_ptr.add(i) as *mut _, res);
        i += 16;
    }

    for j in i..count {
        *acc_ptr.add(j) -= *w_ptr.add(j) as i16;
    }
}

// Scalar Fallbacks

unsafe fn add_feature_scalar(acc: &mut [i16], weights: &[i16]) {
    for (a, w) in acc.iter_mut().zip(weights.iter()) {
        *a += *w;
    }
}

unsafe fn remove_feature_scalar(acc: &mut [i16], weights: &[i16]) {
    for (a, w) in acc.iter_mut().zip(weights.iter()) {
        *a -= *w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_new() {
        let acc = Accumulator::<128>::new();
        assert_eq!(acc.accumulation[0].len(), 128);
        assert_eq!(acc.accumulation[1].len(), 128);
        assert_eq!(acc.computed, [false, false]);
    }
}
