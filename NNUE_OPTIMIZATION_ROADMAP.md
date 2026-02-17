# NNUE Optimization Roadmap

## Overview
This document tracks the optimization progress for the `nnuebie` crate, a Rust implementation of Stockfish's NNUE.
Comparison against Stockfish 17.1 codebase (`archive/nnue`).

## 🚨 Critical Issues (High Priority)

### 1. Fix Finny Table (AccumulatorCache) Updates
**Current Status:** 🔴 Broken / Inefficient
- **Problem:** `finny_tables.rs` uses `Vec<PieceInfo>` allocation inside `compute_diff` and `try_refresh`. This heap allocation in the hot path is a major performance killer.
- **Problem:** The cache update logic (`update_cache`) copies the *current* accumulator back to the cache, but `try_refresh` only reads. Stockfish updates the cache entry *in-place* efficiently while applying updates.
- **Action:**
    - Rewrite `AccumulatorCacheEntry` to use fixed-size arrays or stack buffers for diffs (remove `Vec`).
    - Implement in-place cache updates similar to Stockfish's `update_accumulator_refresh_cache`. The cache entry itself should be the "source of truth" that gets updated and then copied/used.

### 2. Integrate Finny Tables into AccumulatorStack
**Current Status:** 🔴 Missing / Logic Error
- **Problem:** `AccumulatorStack::update_incremental` blindly chains incremental updates even if the King has moved.
- **Impact:** `Accumulator::update_incremental` assumes constant King square. If King moves, the index calculation is wrong, leading to incorrect evaluations or panic.
- **Action:**
    - Modify `AccumulatorStack::update_incremental` to detect King moves.
    - If King moves:
        1. Try `FinnyTables::try_refresh` (which should return the cached value + incremental updates).
        2. If cache miss, fallback to full `refresh`.
    - Only use `Accumulator::update_incremental` for non-King moves.

## 🚀 Optimizations

### 3. VNNI & AVX512 Support for Affine Layers
**Current Status:** ⚠️ Partial (AVX2 simulation)
- **Problem:** `layers.rs` uses `_mm256_maddubs_epi16` + `_mm256_madd_epi16` for AVX2. This is good for AVX2 but suboptimal for CPUs with native VNNI (`vpdpbusd`).
- **Action:**
    - Add `simd_avx512` feature path using `_mm512_dpbusd_epi32` (or `_mm256_dpbusd_epi32` for AVX-VNNI).
    - Ensure weights are permuted correctly to match the intrinsic's expected memory layout.

### 4. Weight Permutation for Affine Transform
**Current Status:** ❓ Suspicious
- **Problem:** Stockfish scrambles weights (`get_weight_index_scrambled`) to optimize for SIMD loading patterns. `nnuebie` reads weights linearly.
- **Action:**
    - Investigate if the current linear weight loading in `AffineTransform` causes cache line splits or inefficient gathering.
    - Implement weight permutation ("scrambling") during model load time if it matches the SIMD kernel better.

### 5. Remove Heap Allocations in Hot Paths
**Current Status:** ⚠️ Needs Cleanup
- **Problem:** usage of `Vec` in `finny_tables.rs` and potentially other helper functions.
- **Action:** Audit codebase for any `Vec::new` or `collect()` inside `evaluate`, `refresh`, or `update` paths. Replace with `ArrayVec` or scratch buffers.

## ✅ Completed

- **Phase 1.1**: Memory Alignment (`AlignedBuffer`) - DONE
- **Phase 1.3**: Weight Permutation (FeatureTransformer) - DONE
- **Phase 1.4**: Single-Pass Accumulator Update - DONE
- **Phase 2.1**: Tile-based Register Accumulation (Refresh) - DONE

## Architectural Differences (vs Stockfish)

| Feature | Stockfish | nnuebie (Current) |
|---------|-----------|-------------------|
| **Accumulator Stack** | Reference-based, no copy on push. Chained updates. | Chained updates, but ignores King moves (Bug). |
| **Accumulator Cache** | In-place update of cache entry. Stack allocs for diffs. | Copy-on-write style. Heap allocs for diffs. |
| **VNNI** | Native `vpdpbusd` (AVX512/VNNI). | AVX2 sequence (`maddubs` + `madd`). |
| **Weights** | Scrambled for SIMD efficiency. | Linear / Output-major (needs verification). |
| **Refresh** | Tiled (Registers). | Tiled (Registers). |
