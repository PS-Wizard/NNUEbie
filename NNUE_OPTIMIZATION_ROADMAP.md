# NNUE Performance Optimization Roadmap

## Overview

This document outlines the performance optimization phases for the nnuebie Rust implementation of Stockfish's NNUE neural network evaluation.

## Current Status

- **Phase 1.1**: Memory Alignment - DONE ✅
- **Phase 1.2**: Finny Tables (AccumulatorCaches) - PARTIAL ⚠️ (Cache not updating)
- **Phase 1.3**: Weight Permutation - DONE ✅
- **Phase 1.4**: Single-Pass Accumulator Update - DONE ✅
- **Phase 1.5**: Fix Finny Table Cache Updates - **PENDING (HIGH)** 🔴
- **Phase 2.1**: Tile-based Register Accumulation (Refresh) - DONE ✅
- **Phase 2.2**: VNNI for MLP Layers - PENDING
- **Phase 2.3**: Pre-populate Finny Tables Cache - PENDING

## Performance Analysis & Comparison (Stockfish vs nnuebie)

| Benchmark | Previous | Current (Phase 2.1) | Stockfish (Ref) |
|-----------|----------|---------------------|-----------------|
| **Full Refresh** | ~191K/s | **~228K/s** | ~400K/s |
| **Incremental** | ~1.33M/s | **~1.36M/s** | ~3M/s |

### Phase 2.1: Tiled Refresh Summary

**Implemented:**
- Added `src/accumulator_refresh.rs` with `refresh_avx2_3072` and `refresh_avx2_128`.
- These kernels use **tiling**: they iterate over accumulator chunks (e.g. 256 dims) and process ALL pieces for that chunk while keeping the accumulator in AVX2 registers.
- This reduces memory read/write traffic by ~32x compared to the previous loop (which iterated pieces first).
- Integrated into `Accumulator::refresh` with runtime detection.

**Result:**
- **Full Refresh throughput increased by ~19%** (191K -> 228K).
- This directly addresses the 61% bottleneck seen in the profiler for `add_feature_avx2` during `set_position`.

### Next Steps

The next critical task is **Phase 1.5: Fix Finny Table Cache Updates**.
Currently, the cache is read-only. We need to implement efficient *cache updates* (using the `AccumulatorCache` struct) so that King moves don't trigger full refreshes at all in real search.

---

## Optimization Phases (Remaining)

### Phase 1.5: Fix Finny Table Cache Updates (HIGH PRIORITY)
**Goal**: Keep Finny Tables efficient by updating cache entries.
**Action**:
- In `try_refresh` (or `refresh_with_cache`), update the *cache entry* in-place using the diffs.
- Then copy the updated cache entry to the current accumulator (or do it in one pass like Stockfish).
- **Expected Impact**: Prevent cache degradation, faster king moves.

### Phase 2: Advanced Optimizations

#### Phase 2.2: VNNI for MLP Layers
**Goal**: Faster neural network inference.
**Action**:
- Use `_mm256_dpbusd_epi32` for affine transform.
- Requires weight scrambling.

#### Phase 2.3: Pre-populate Finny Tables Cache
**Goal**: Instant start for any king position.
**Action**:
- At initialization, compute full accumulators for all 64 king squares with all pieces.
- Store in Finny Tables.
- **Reference**: Stockfish `AccumulatorCaches::clear`.

---

## Implementation References

### Stockfish Logic (Single Pass Update)
`nnue_accumulator.cpp`:
```cpp
// Iterate over accumulator in tiles (e.g. 4 registers)
for (IndexType i = 0; i < Dimensions / TileHeight; ++i) {
    // Load accumulator tile
    vec_t acc[NumRegs];
    // ... load acc ...

    // Apply ALL updates to this tile
    for (const auto& added : added_list) {
         acc[k] = vec_add_16(acc[k], weights[added]);
    }
    for (const auto& removed : removed_list) {
         acc[k] = vec_sub_16(acc[k], weights[removed]);
    }

    // Store accumulator tile
    // ... store acc ...
}
```
