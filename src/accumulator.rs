use crate::feature_transformer::FeatureTransformer;
use crate::features::{self, make_index};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone)]
pub struct Accumulator {
    pub accumulation: [Vec<i16>; 2],
    pub psqt_accumulation: [Vec<i32>; 2], // Size 8
    pub computed: [bool; 2],
}

impl Accumulator {
    pub fn new(size: usize) -> Self {
        Self {
            accumulation: [vec![0; size], vec![0; size]],
            psqt_accumulation: [vec![0; 8], vec![0; 8]],
            computed: [false, false],
        }
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
        // Reset
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

        // Vectorized update
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                let acc_ptr = self.accumulation[perspective].as_mut_ptr();
                let w_ptr = w_slice.as_ptr();
                let count = half_dims;

                // Process 16 elements at a time
                for _ in 0..(count / 16) {
                    let w = _mm256_loadu_si256(w_ptr.add(i) as *const _);
                    let a = _mm256_loadu_si256(acc_ptr.add(i) as *const _);
                    let res = _mm256_sub_epi16(a, w);
                    _mm256_storeu_si256(acc_ptr.add(i) as *mut _, res);
                    i += 16;
                }

                // Handle remainder
                for j in i..count {
                    *acc_ptr.add(j) -= *w_ptr.add(j) as i16;
                }
            }
        } else {
            // Scalar fallback
            for (i, &w) in w_slice.iter().enumerate() {
                self.accumulation[perspective][i] -= w;
            }
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
        let _input_dims = ft.input_dims; // Actually padded_input_dims if we followed C++ exactly
        let offset = feature_idx * half_dims;
        let w_slice = &ft.weights[offset..offset + half_dims];

        // Vectorized update
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                let acc_ptr = self.accumulation[perspective].as_mut_ptr();
                let w_ptr = w_slice.as_ptr();
                let count = half_dims;

                // Process 16 elements at a time
                for _ in 0..(count / 16) {
                    let w = _mm256_loadu_si256(w_ptr.add(i) as *const _);
                    let a = _mm256_loadu_si256(acc_ptr.add(i) as *const _);
                    let res = _mm256_add_epi16(a, w);
                    _mm256_storeu_si256(acc_ptr.add(i) as *mut _, res);
                    i += 16;
                }

                // Handle remainder
                for j in i..count {
                    *acc_ptr.add(j) += *w_ptr.add(j) as i16;
                }
            }
        } else {
            // Scalar fallback
            for (i, &w) in w_slice.iter().enumerate() {
                self.accumulation[perspective][i] += w;
            }
        }

        // PSQT update
        // psqt_weights: [Input][Buckets]
        // psqtWeights size `InputDimensions * PSQTBuckets`.
        let psqt_offset = feature_idx * crate::feature_transformer::PSQT_BUCKETS;
        let psqt_slice =
            &ft.psqt_weights[psqt_offset..psqt_offset + crate::feature_transformer::PSQT_BUCKETS];

        for (i, &pw) in psqt_slice.iter().enumerate() {
            self.psqt_accumulation[perspective][i] += pw;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_new() {
        let size = 128;
        let acc = Accumulator::new(size);
        assert_eq!(acc.accumulation[0].len(), size);
        assert_eq!(acc.accumulation[1].len(), size);
        assert_eq!(acc.computed, [false, false]);
    }
}
