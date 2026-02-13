use crate::feature_transformer::FeatureTransformer;
use crate::features::{self, make_index};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

type FeatureUpdateFn = unsafe fn(&mut [i16], &[i16]);

#[derive(Clone)]
pub struct Accumulator<const SIZE: usize> {
    // Use heap-allocated aligned memory
    pub accumulation: [Box<[i16; SIZE]>; 2],
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
        let acc0 = Box::new([0i16; SIZE]);
        let acc1 = Box::new([0i16; SIZE]);

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

    fn remove_feature(&mut self, perspective: usize, feature_idx: usize, ft: &FeatureTransformer) {
        let half_dims = ft.half_dims;
        let offset = feature_idx * half_dims;
        let w_slice = &ft.weights[offset..offset + half_dims];

        // Use selected implementation
        unsafe {
            (self.remove_feature_fn)(self.accumulation[perspective].as_mut_slice(), w_slice);
        }

        let psqt_offset = feature_idx * crate::feature_transformer::PSQT_BUCKETS;
        let psqt_slice =
            &ft.psqt_weights[psqt_offset..psqt_offset + crate::feature_transformer::PSQT_BUCKETS];

        for (i, &pw) in psqt_slice.iter().enumerate() {
            self.psqt_accumulation[perspective][i] -= pw;
        }
    }

    fn add_feature(&mut self, perspective: usize, feature_idx: usize, ft: &FeatureTransformer) {
        let half_dims = ft.half_dims;
        let offset = feature_idx * half_dims;
        let w_slice = &ft.weights[offset..offset + half_dims];

        // Use selected implementation
        unsafe {
            (self.add_feature_fn)(self.accumulation[perspective].as_mut_slice(), w_slice);
        }

        // PSQT update
        let psqt_offset = feature_idx * crate::feature_transformer::PSQT_BUCKETS;
        let psqt_slice =
            &ft.psqt_weights[psqt_offset..psqt_offset + crate::feature_transformer::PSQT_BUCKETS];

        for (i, &pw) in psqt_slice.iter().enumerate() {
            self.psqt_accumulation[perspective][i] += pw;
        }
    }
}

// SIMD Implementations

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_feature_avx2(acc: &mut [i16], weights: &[i16]) {
    let mut i = 0;
    let acc_ptr = acc.as_mut_ptr();
    let w_ptr = weights.as_ptr();
    let count = acc.len();

    // Unroll by 4 (64 elements per iteration)
    // Use unaligned loads for safety - Box doesn't guarantee alignment
    while i + 64 <= count {
        let w0 = _mm256_loadu_si256(w_ptr.add(i) as *const _);
        let w1 = _mm256_loadu_si256(w_ptr.add(i + 16) as *const _);
        let w2 = _mm256_loadu_si256(w_ptr.add(i + 32) as *const _);
        let w3 = _mm256_loadu_si256(w_ptr.add(i + 48) as *const _);

        let a0 = _mm256_loadu_si256(acc_ptr.add(i) as *const _);
        let a1 = _mm256_loadu_si256(acc_ptr.add(i + 16) as *const _);
        let a2 = _mm256_loadu_si256(acc_ptr.add(i + 32) as *const _);
        let a3 = _mm256_loadu_si256(acc_ptr.add(i + 48) as *const _);

        let r0 = _mm256_add_epi16(a0, w0);
        let r1 = _mm256_add_epi16(a1, w1);
        let r2 = _mm256_add_epi16(a2, w2);
        let r3 = _mm256_add_epi16(a3, w3);

        _mm256_storeu_si256(acc_ptr.add(i) as *mut _, r0);
        _mm256_storeu_si256(acc_ptr.add(i + 16) as *mut _, r1);
        _mm256_storeu_si256(acc_ptr.add(i + 32) as *mut _, r2);
        _mm256_storeu_si256(acc_ptr.add(i + 48) as *mut _, r3);

        i += 64;
    }

    // Remainder loop (if size not multiple of 64)
    while i + 16 <= count {
        let w = _mm256_loadu_si256(w_ptr.add(i) as *const _);
        let a = _mm256_loadu_si256(acc_ptr.add(i) as *const _);
        let res = _mm256_add_epi16(a, w);
        _mm256_storeu_si256(acc_ptr.add(i) as *mut _, res);
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
        let w0 = _mm256_loadu_si256(w_ptr.add(i) as *const _);
        let w1 = _mm256_loadu_si256(w_ptr.add(i + 16) as *const _);
        let w2 = _mm256_loadu_si256(w_ptr.add(i + 32) as *const _);
        let w3 = _mm256_loadu_si256(w_ptr.add(i + 48) as *const _);

        let a0 = _mm256_loadu_si256(acc_ptr.add(i) as *const _);
        let a1 = _mm256_loadu_si256(acc_ptr.add(i + 16) as *const _);
        let a2 = _mm256_loadu_si256(acc_ptr.add(i + 32) as *const _);
        let a3 = _mm256_loadu_si256(acc_ptr.add(i + 48) as *const _);

        let r0 = _mm256_sub_epi16(a0, w0);
        let r1 = _mm256_sub_epi16(a1, w1);
        let r2 = _mm256_sub_epi16(a2, w2);
        let r3 = _mm256_sub_epi16(a3, w3);

        _mm256_storeu_si256(acc_ptr.add(i) as *mut _, r0);
        _mm256_storeu_si256(acc_ptr.add(i + 16) as *mut _, r1);
        _mm256_storeu_si256(acc_ptr.add(i + 32) as *mut _, r2);
        _mm256_storeu_si256(acc_ptr.add(i + 48) as *mut _, r3);

        i += 64;
    }

    while i + 16 <= count {
        let w = _mm256_loadu_si256(w_ptr.add(i) as *const _);
        let a = _mm256_loadu_si256(acc_ptr.add(i) as *const _);
        let res = _mm256_sub_epi16(a, w);
        _mm256_storeu_si256(acc_ptr.add(i) as *mut _, res);
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
