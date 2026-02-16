# NNUE Performance Optimization Roadmap

## Overview

This document outlines the performance optimization phases for the nnuebie Rust implementation of Stockfish's NNUE neural network evaluation.

## Current Status

- **Phase 1.1**: Memory Alignment - DONE
- **Phase 1.2**: Finny Tables (AccumulatorCaches) - DONE
- **Phase 1.3**: Weight Permutation - DONE ✅
- **Phase 1.4**: Accumulator Deep Copy Fix - **NOT IMPLEMENTED** ❌
- **Phase 2.1**: Tile-based Register Accumulation - PENDING
- **Phase 2.2**: VNNI for MLP Layers - PENDING
- **Phase 2.3**: Pre-populate Finny Tables Cache - PENDING

## Current Performance

| Benchmark | Current | Stockfish (ref) | Target |
|-----------|---------|-----------------|--------|
| Full Refresh | ~218K/s | ~400K/s | 400K/s |
| Incremental Update | ~1.21M/s | ~3M/s | 2.5M/s |
| Speedup Ratio | 5.54x | ~7x | - |

### Hot Path Analysis (perf report)

```
65.64%  add_feature_avx2        # PRIMARY BOTTLENECK
13.37%  AffineTransform::propagate
 6.13%  NNUEProbe::evaluate
 4.42%  refresh_accumulators
 3.03%  remove_feature_avx2
```

**Key Insight**: `add_feature_avx2` dominates runtime. Weight permutation provided minimal benefit because the feature transformer is only 6% of runtime. The real bottleneck is the accumulator update (add/remove features).

---

## Phase 1.3 Deep Dive: Weight Permutation (COMPLETED)

### What Was Done

1. **Added permutation function** to `feature_transformer.rs`
2. **Applied PackusEpi16Order** pattern at load time: `[0, 2, 1, 3, 4, 6, 5, 7]`
3. **Rewrote transform_features_avx2** to use:
   - Interleaved access: `j*2` and `j*2+1`
   - Asymmetric clipping (max on first operand only)
   - Direct `_mm256_packus_epi16` to combine and clip

### Implementation Details

```rust
// feature_transformer.rs
const PACKUS_EPI16_ORDER: [usize; 8] = [0, 2, 1, 3, 4, 6, 5, 7];

fn permute_weights(data: &mut [i16]) {
    // Reorder 16-element blocks according to PackusEpi16Order
}
```

```rust
// network.rs - transform_features_avx2
// Process 32 output elements per iteration
for j in (0..n).step_by(32) {
    let v0a = _mm256_load_si256(acc_ptr.add(j));
    let v0b = _mm256_load_si256(acc_ptr.add(j + 16));
    // Asymmetric clipping
    let v0a_c = _mm256_max_epi16(max, _mm256_min_epi16(min, v0a));
    let v1a_c = _mm256_min_epi16(min, v1a);
    // mulhi + packus
    let packed = _mm256_packus_epi16(pa, pb);
}
```

### Results

- ✅ All validation tests pass (exact match with Stockfish)
- ✅ Modest ~4% improvement in incremental throughput
- ✅ Reduced instruction count in feature transformer

### Why Limited Impact

The feature transformer is only ~6% of total runtime. The major bottleneck is `add_feature_avx2` (65% of runtime), which was not addressed by weight permutation.

---

## Phase 1.4 Deep Dive: Accumulator Deep Copy Fix (NOT IMPLEMENTED ❌)

### The Problem

In `accumulator_stack.rs:182-202`, the code STILL copies entire accumulators:

```rust
// Lines 182-188 - STILL COPYING!
let source_big: Option<(Vec<i16>, Vec<i16>, [[i32; 8]; 2])> = if has_computed {
    let source = &self.stack[last_computed_idx];
    Some((
        source.acc_big.accumulation[0].as_slice().to_vec(),  // COPY! ~6KB
        source.acc_big.accumulation[1].as_slice().to_vec(),  // COPY! ~6KB
        source.acc_big.psqt_accumulation,
    ))
} else {
    None
};
```

Despite the comment at line 136 claiming "Stockfish approach: NO copying!", the code uses `.to_vec()` which allocates and copies ~12KB on every incremental update.

### What Stockfish Actually Does

1. **`push()`** - literally does NOTHING except mark state as dirty
2. **At evaluation time** - finds last computed state, then:
   - Uses shared memory (no copy needed)
   - Updates forward from source state in-place

### Required Fix

Replace `.to_vec()` copies with direct slice copies, or better - avoid copying entirely by updating in-place from source.

### Impact

- Each `update_incremental` copies ~12KB (big) + ~256 bytes (small)
- For search with 100K+ nodes, this adds significant overhead

---

## Phase 2: Next Optimizations

### Phase 2.1: Tile-based Register Accumulation (HIGH PRIORITY)

**Problem**: Current implementation loads 4 vectors (64 elements), adds, stores immediately. This causes high memory bandwidth usage.

**Stockfish's Approach** (nnue_accumulator.cpp:266-318):
- Load 8+ SIMD registers at once (TileHeight = 128 bytes = 8 AVX2 registers)
- Accumulate in registers across all pieces
- Store once at the end

**Implementation**:
```rust
// Instead of: add_feature per piece, store each time
// Do: accumulate in registers, store once

unsafe fn add_feature_avx2_batch(acc: &mut [i16], weights: &[&[i16]]) {
    // Load all feature columns into registers first
    // Then add to accumulator registers
    // Finally store once
}
```

**Expected Impact**: 30-50% speedup in add_feature (biggest win)

### Phase 2.2: VNNI for MLP Layers (MEDIUM PRIORITY)

**Problem**: Current affine transform uses:
```rust
let p0 = _mm256_maddubs_epi16(input_vec, w0);
let s0 = _mm256_madd_epi16(p0, ones);
acc0 = _mm256_add_epi32(acc0, s0);
```

**Stockfish's VNNI path**:
```cpp
acc = _mm256_dpbusd_epi32(acc, a, b);  // Single instruction!
```

**Required**: Weight scrambling for VNNI compatibility

**Expected Impact**: 15-25% speedup in MLP propagation

### Phase 2.3: Optimize Refresh with Cached Delta

**Problem**: Even with Finny Tables, refresh does full add/remove operations.

**Stockfish's Approach**:
- Uses cached entry directly with delta computation
- Single fused add-sub operation per piece
- Batched PSQT updates

**Expected Impact**: 20-30% speedup in refresh scenarios

### Phase 2.4: Pre-populate Finny Tables Cache (HIGH PRIORITY)

**Problem**: Current Finny Tables cache starts with BIASES only (no pieces).

**Stockfish's Approach**:
- At initialization, for EACH king square (64 positions):
  - Compute FULL accumulator with ALL pieces
  - Store in cache entry
- On king move: COPY from cache (instant!)

**Current Implementation**:
```rust
pub fn clear(&mut self, biases: &[i16]) {
    for square_entries in &mut self.entries {
        for entry in square_entries {
            entry.clear(biases);  // Only biases, no pieces!
        }
    }
}
```

**Required**: Pre-populate with full piece sets for each king position.

**Expected Impact**: 50%+ speedup on king moves (most common refresh scenario)

---

## Finny Tables (Phase 1.2) - Implementation Summary

### What Was Implemented

- `AccumulatorCacheEntry<SIZE>`: Cached state per (king_square, perspective)
- `AccumulatorCache<SIZE>`: 64 entries × 2 perspectives
- `FinnyTables`: Wrapper for big (3072) and small (128) networks
- `try_refresh()`: Attempt fast refresh from cache
- `update_cache()`: Update cache after full refresh

### How It Works

1. **Initialization**: Cache entries cleared with biases (no pieces)
2. **On King Move**: 
   - Try `try_refresh()` - if cache is valid and diff ≤ 4 pieces
   - Copy cached accumulator + apply piece differences
   - Much faster than full refresh
3. **On Full Refresh**: Update cache with new state

### Current Limitations

- `MAX_DIFF_PIECES = 4` - only works for small changes
- Cache invalidation not fully utilized
- Does NOT pre-compute per-king-square states at init (Stockfish does this)

### Stockfish's Better Approach

Stockfish pre-populates cache at initialization:
- For each king square: compute accumulator with ALL pieces
- Store in cache entry
- On king move: copy from cache (already computed!)

**This is the BIGGEST missing optimization** - we compute from biases each time.

---

## Architecture Notes

### Key Stockfish Files for Reference
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_common.h` - Constants (CacheLineSize=64)
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_accumulator.h` - Accumulator + Cache
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_accumulator.cpp` - Implementation (line 266-318 for SIMD update)
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_feature_transformer.h` - Feature transformer + permutation
- `archive/nnue/Stockfish-sf_17.1/src/nnue/layers/affine_transform.h` - MLP layers with VNNI
- `archive/nnue/Stockfish-sf_17.1/src/nnue/layers/simd.h` - SIMD definitions (lines 68-77 for VNNI)

### Memory Alignment Requirements
- All SIMD data: 64-byte aligned
- Use `_mm256_load_si256` (not `_mm256_loadu_si256`) for aligned memory
- Cache line size: 64 bytes

### SIMD Instructions Used
- AVX2: `_mm256_dpbusd_epi32` (VNNI dot product - single instruction!)
- AVX2: `_mm256_maddubs_epi16` (byte * signed byte -> word)
- AVX2: `_mm256_madd_epi16` (word * word -> dword)
- AVX2: `_mm256_mulhi_epi16` (high 16 bits of 16-bit multiply - faster!)
- AVX512: `_mm512_dpbusd_epi32` (512-bit VNNI)

### PackusEpi16Order Pattern
Stockfish uses this permutation for optimal SIMD:
```cpp
// AVX2
return {0, 2, 1, 3, 4, 6, 5, 7};
// AVX512
return {0, 2, 4, 6, 1, 3, 5, 7};
```

### Weight Scrambling for VNNI
For MLP layers, Stockfish scrambles weights to enable VNNI:
```cpp
get_weight_index_scrambled(i) = 
    (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4
    + i / PaddedInputDimensions * 4 + i % 4;
```
