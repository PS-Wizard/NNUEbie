use crate::aligned::AlignedBuffer;
use crate::feature_transformer::FeatureTransformer;
use crate::features::{self, make_index};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Type for the single-pass update function
// prev_acc, curr_acc, added_weights, removed_weights
type UpdateSinglePassFn = unsafe fn(&[i16], &mut [i16], &[&[i16]], &[&[i16]]);

type FeatureUpdateFn = unsafe fn(&mut [i16], &[i16]);

// Type for tiled refresh function
type RefreshFn = unsafe fn(&mut [i16], &[i16], &AlignedBuffer<i16>, &[usize]);

#[derive(Clone)]
pub struct Accumulator<const SIZE: usize> {
    // Use heap-allocated aligned memory
    pub accumulation: [AlignedBuffer<i16>; 2],
    pub psqt_accumulation: [[i32; 8]; 2],
    pub computed: [bool; 2],
    add_feature_fn: FeatureUpdateFn,
    remove_feature_fn: FeatureUpdateFn,
    update_single_pass_fn: UpdateSinglePassFn,
    refresh_fn: Option<RefreshFn>,
}

impl<const SIZE: usize> Default for Accumulator<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SIZE: usize> Accumulator<SIZE> {
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        let (add_fn, remove_fn, update_fn, refresh_fn) = if is_x86_feature_detected!("avx2") {
            let r_fn = if SIZE == 3072 {
                Some(crate::accumulator_refresh::refresh_avx2_3072 as RefreshFn)
            } else if SIZE == 128 {
                Some(crate::accumulator_refresh::refresh_avx2_128 as RefreshFn)
            } else {
                None
            };

            (
                add_feature_avx2 as FeatureUpdateFn,
                remove_feature_avx2 as FeatureUpdateFn,
                update_accumulators_single_pass_avx2 as UpdateSinglePassFn,
                r_fn,
            )
        } else {
            (
                add_feature_scalar as FeatureUpdateFn,
                remove_feature_scalar as FeatureUpdateFn,
                update_accumulators_single_pass_scalar as UpdateSinglePassFn,
                None,
            )
        };

        #[cfg(not(target_arch = "x86_64"))]
        let (add_fn, remove_fn, update_fn, refresh_fn) = (
            add_feature_scalar as FeatureUpdateFn,
            remove_feature_scalar as FeatureUpdateFn,
            update_accumulators_single_pass_scalar as UpdateSinglePassFn,
            None,
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
            update_single_pass_fn: update_fn,
            refresh_fn,
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

        // If we have an optimized tiled refresh kernel, use it
        if let Some(refresh_kernel) = self.refresh_fn {
            // Collect feature indices first
            // We use a Vec here, but in a real engine we'd use a small vec or scratch buffer
            // Max 32 pieces
            let mut indices_w: [usize; 32] = [0; 32];
            let mut indices_b: [usize; 32] = [0; 32];
            let mut count = 0;

            for (i, &(sq, pc)) in pieces.iter().enumerate().take(32) {
                indices_w[i] = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
                indices_b[i] = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
                count += 1;
            }

            unsafe {
                // White
                refresh_kernel(
                    self.accumulation[0].as_mut_slice(),
                    &ft.biases,
                    &ft.weights,
                    &indices_w[..count],
                );
                // Black
                refresh_kernel(
                    self.accumulation[1].as_mut_slice(),
                    &ft.biases,
                    &ft.weights,
                    &indices_b[..count],
                );
            }

            // Still need to do PSQT (scalar/simd separate path)
            self.psqt_accumulation[0].fill(0);
            self.psqt_accumulation[1].fill(0);
            for &(sq, pc) in pieces {
                let idx_w = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
                self.update_psqt(features::WHITE, idx_w, ft, true);

                let idx_b = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
                self.update_psqt(features::BLACK, idx_b, ft, true);
            }

            self.computed = [true, true];
            return;
        }

        // Fallback: Scalar / Non-tiled loop
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

    /// Incrementally updates the accumulator from a previous state.
    ///
    /// This combines copy, add, and remove into a single pass for efficiency.
    ///
    /// This function assumes that the King Squares (`ksq`) have NOT changed for the perspectives being updated.
    /// If a King has moved for a given color, you must use `refresh` for that color's accumulator instead.
    pub fn update_incremental(
        &mut self,
        prev: &Accumulator<SIZE>,
        added: &[(usize, usize)],   // (Square, Piece)
        removed: &[(usize, usize)], // (Square, Piece)
        ksq: [usize; 2],
        ft: &FeatureTransformer,
    ) {
        debug_assert_eq!(
            ft.half_dims, SIZE,
            "FeatureTransformer dims mismatch Accumulator size"
        );

        // Update PSQT first (still separate pass for now, but small cost)
        // Copy PSQT from previous
        self.psqt_accumulation = prev.psqt_accumulation;

        for &(sq, pc) in removed {
            let idx_w = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
            self.update_psqt(features::WHITE, idx_w, ft, false);

            let idx_b = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
            self.update_psqt(features::BLACK, idx_b, ft, false);
        }
        for &(sq, pc) in added {
            let idx_w = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
            self.update_psqt(features::WHITE, idx_w, ft, true);

            let idx_b = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
            self.update_psqt(features::BLACK, idx_b, ft, true);
        }

        // Prepare weight slices using stack arrays to avoid heap allocation
        // Max 3 changed pieces per move
        let mut added_weights_w: [&[i16]; 3] = [&[]; 3];
        let mut removed_weights_w: [&[i16]; 3] = [&[]; 3];

        for (i, &(sq, pc)) in added.iter().enumerate().take(3) {
            let idx = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
            let offset = idx * SIZE;
            added_weights_w[i] = &ft.weights[offset..offset + SIZE];
        }
        for (i, &(sq, pc)) in removed.iter().enumerate().take(3) {
            let idx = make_index(features::WHITE, sq, pc, ksq[features::WHITE]);
            let offset = idx * SIZE;
            removed_weights_w[i] = &ft.weights[offset..offset + SIZE];
        }

        // Single pass update for White
        unsafe {
            (self.update_single_pass_fn)(
                &prev.accumulation[features::WHITE],
                &mut self.accumulation[features::WHITE],
                &added_weights_w[..added.len()],
                &removed_weights_w[..removed.len()],
            );
        }

        // Prepare weight slices for Black Perspective
        let mut added_weights_b: [&[i16]; 3] = [&[]; 3];
        let mut removed_weights_b: [&[i16]; 3] = [&[]; 3];

        for (i, &(sq, pc)) in added.iter().enumerate().take(3) {
            let idx = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
            let offset = idx * SIZE;
            added_weights_b[i] = &ft.weights[offset..offset + SIZE];
        }
        for (i, &(sq, pc)) in removed.iter().enumerate().take(3) {
            let idx = make_index(features::BLACK, sq, pc, ksq[features::BLACK]);
            let offset = idx * SIZE;
            removed_weights_b[i] = &ft.weights[offset..offset + SIZE];
        }

        // Single pass update for Black
        unsafe {
            (self.update_single_pass_fn)(
                &prev.accumulation[features::BLACK],
                &mut self.accumulation[features::BLACK],
                &added_weights_b[..added.len()],
                &removed_weights_b[..removed.len()],
            );
        }

        self.computed = [true, true];
    }

    /// Legacy update method (still needed for Finny Tables fallback etc)
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

// Single-pass update AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_accumulators_single_pass_avx2(
    prev_acc: &[i16],
    curr_acc: &mut [i16],
    added_weights: &[&[i16]],
    removed_weights: &[&[i16]],
) {
    let mut i = 0;
    let prev_ptr = prev_acc.as_ptr();
    let curr_ptr = curr_acc.as_mut_ptr();
    let count = prev_acc.len();

    // Specialization for common case: 1 added, 1 removed (Normal move / Capture)
    if added_weights.len() == 1 && removed_weights.len() == 1 {
        let w_add = added_weights[0].as_ptr();
        let w_rem = removed_weights[0].as_ptr();

        while i + 64 <= count {
            let a0 = _mm256_load_si256(prev_ptr.add(i) as *const _);
            let a1 = _mm256_load_si256(prev_ptr.add(i + 16) as *const _);
            let a2 = _mm256_load_si256(prev_ptr.add(i + 32) as *const _);
            let a3 = _mm256_load_si256(prev_ptr.add(i + 48) as *const _);

            let ra0 = _mm256_load_si256(w_rem.add(i) as *const _);
            let ra1 = _mm256_load_si256(w_rem.add(i + 16) as *const _);
            let ra2 = _mm256_load_si256(w_rem.add(i + 32) as *const _);
            let ra3 = _mm256_load_si256(w_rem.add(i + 48) as *const _);

            let aa0 = _mm256_load_si256(w_add.add(i) as *const _);
            let aa1 = _mm256_load_si256(w_add.add(i + 16) as *const _);
            let aa2 = _mm256_load_si256(w_add.add(i + 32) as *const _);
            let aa3 = _mm256_load_si256(w_add.add(i + 48) as *const _);

            let r0 = _mm256_add_epi16(_mm256_sub_epi16(a0, ra0), aa0);
            let r1 = _mm256_add_epi16(_mm256_sub_epi16(a1, ra1), aa1);
            let r2 = _mm256_add_epi16(_mm256_sub_epi16(a2, ra2), aa2);
            let r3 = _mm256_add_epi16(_mm256_sub_epi16(a3, ra3), aa3);

            _mm256_store_si256(curr_ptr.add(i) as *mut _, r0);
            _mm256_store_si256(curr_ptr.add(i + 16) as *mut _, r1);
            _mm256_store_si256(curr_ptr.add(i + 32) as *mut _, r2);
            _mm256_store_si256(curr_ptr.add(i + 48) as *mut _, r3);

            i += 64;
        }
    } else {
        // Generic path unrolled 4x
        while i + 64 <= count {
            let mut a0 = _mm256_load_si256(prev_ptr.add(i) as *const _);
            let mut a1 = _mm256_load_si256(prev_ptr.add(i + 16) as *const _);
            let mut a2 = _mm256_load_si256(prev_ptr.add(i + 32) as *const _);
            let mut a3 = _mm256_load_si256(prev_ptr.add(i + 48) as *const _);

            for w_slice in removed_weights {
                let ptr = w_slice.as_ptr();
                let w0 = _mm256_load_si256(ptr.add(i) as *const _);
                let w1 = _mm256_load_si256(ptr.add(i + 16) as *const _);
                let w2 = _mm256_load_si256(ptr.add(i + 32) as *const _);
                let w3 = _mm256_load_si256(ptr.add(i + 48) as *const _);
                a0 = _mm256_sub_epi16(a0, w0);
                a1 = _mm256_sub_epi16(a1, w1);
                a2 = _mm256_sub_epi16(a2, w2);
                a3 = _mm256_sub_epi16(a3, w3);
            }

            for w_slice in added_weights {
                let ptr = w_slice.as_ptr();
                let w0 = _mm256_load_si256(ptr.add(i) as *const _);
                let w1 = _mm256_load_si256(ptr.add(i + 16) as *const _);
                let w2 = _mm256_load_si256(ptr.add(i + 32) as *const _);
                let w3 = _mm256_load_si256(ptr.add(i + 48) as *const _);
                a0 = _mm256_add_epi16(a0, w0);
                a1 = _mm256_add_epi16(a1, w1);
                a2 = _mm256_add_epi16(a2, w2);
                a3 = _mm256_add_epi16(a3, w3);
            }

            _mm256_store_si256(curr_ptr.add(i) as *mut _, a0);
            _mm256_store_si256(curr_ptr.add(i + 16) as *mut _, a1);
            _mm256_store_si256(curr_ptr.add(i + 32) as *mut _, a2);
            _mm256_store_si256(curr_ptr.add(i + 48) as *mut _, a3);

            i += 64;
        }
    }

    // Remainder loop (16 elements)
    while i + 16 <= count {
        let mut acc = _mm256_load_si256(prev_ptr.add(i) as *const _);

        for w_slice in removed_weights {
            let w = _mm256_load_si256(w_slice.as_ptr().add(i) as *const _);
            acc = _mm256_sub_epi16(acc, w);
        }

        for w_slice in added_weights {
            let w = _mm256_load_si256(w_slice.as_ptr().add(i) as *const _);
            acc = _mm256_add_epi16(acc, w);
        }

        _mm256_store_si256(curr_ptr.add(i) as *mut _, acc);
        i += 16;
    }

    // Scalar fallback for final remainder
    for j in i..count {
        let mut val = *prev_ptr.add(j);
        for w_slice in removed_weights {
            val -= *w_slice.as_ptr().add(j);
        }
        for w_slice in added_weights {
            val += *w_slice.as_ptr().add(j);
        }
        *curr_ptr.add(j) = val;
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

unsafe fn update_accumulators_single_pass_scalar(
    prev_acc: &[i16],
    curr_acc: &mut [i16],
    added_weights: &[&[i16]],
    removed_weights: &[&[i16]],
) {
    let count = prev_acc.len();
    for i in 0..count {
        let mut val = prev_acc[i];
        for w_slice in removed_weights {
            val = val.wrapping_sub(w_slice[i]);
        }
        for w_slice in added_weights {
            val = val.wrapping_add(w_slice[i]);
        }
        curr_acc[i] = val;
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
