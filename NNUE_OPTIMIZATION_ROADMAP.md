# NNUE Performance Optimization Roadmap

## Overview

This document outlines the performance optimization phases for the nnuebie Rust implementation of Stockfish's NNUE neural network evaluation.

## Current Status

- **Phase 1.1**: Memory Alignment - DONE
- **Phase 1.2**: Finny Tables (AccumulatorCaches) - DONE
- **Phase 1.3**: Weight Permutation - **CRITICAL - NOT IMPLEMENTED**
- **Phase 1.4**: Accumulator Deep Copy Fix - DONE (partial implementation)

## Current Performance

| Benchmark | Current | Target |
|-----------|---------|--------|
| Full Refresh | ~220K/s | 400K/s |
| Incremental Update | ~1.18M/s | 2.5M/s |

---

## Critical Issues Found

### 1. Weight Permutation NOT Implemented (Phase 1.3 - CRITICAL)

**Problem**: nnuebie reads weights in standard file order, but Stockfish permutes weights at load time for optimal SIMD behavior.

**Stockfish Approach** (`nnue_feature_transformer.h:289-319`):
```cpp
void permute_weights() {
    permute<16>(biases, PackusEpi16Order);
    permute<16>(weights, PackusEpi16Order);
}

void scale_weights(bool read) {
    for (j) {
        for (i)
            w[i] = read ? w[i] * 2 : w[i] / 2;
    }
}

bool read_parameters(std::istream& stream) {
    // ... read weights
    permute_weights();  // Critical!
    scale_weights(true);
}
```

**What nnuebie does** (`feature_transformer.rs:26-48`):
```rust
pub fn read_parameters(&mut self, reader: &mut R, skip_first_magic: bool) {
    // ... read weights
    // Only scales by 2 - NO PERMUTATION!
    for b in &mut biases_vec { *b = b.wrapping_mul(2); }
    for w in &mut weights_vec { *w = w.wrapping_mul(2); }
}
```

**Expected Impact**: 15-30% improvement in feature transformation

---

### 2. MLP Weight Scrambling NOT Implemented (CRITICAL)

**Problem**: Stockfish uses scrambled weight ordering in affine transforms to enable VNNI dot product instructions.

**Stockfish Approach** (`affine_transform.h:153-164, 188-239`):
```cpp
// Scrambled indexing for VNNI
static constexpr IndexType get_weight_index_scrambled(IndexType i) {
    return (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4
         + i / PaddedInputDimensions * 4 + i % 4;
}

// Uses VNNI dot product
vec_add_dpbusd_32(acc[k], in0, col0[k]);  // _mm256_dpbusd_epi32
```

**nnuebie Approach** (`layers.rs`):
- Uses simple row-major weight storage
- Does NOT use VNNI properly - uses separate maddubs + madd

**Expected Impact**: 10-20% improvement in MLP propagation

---

### 3. Feature Transformer Multiplication Not Optimized

**Problem**: Stockfish uses `mulhi` for faster multiplication. -- DONE

**Stockfish Approach** (`nnue_feature_transformer.h:427-446`):
```cpp
// Scale by 2 at load time allows using mulhi (shift left 6, mulhi = shift right 16, net = 9)
// Dividing by 128 * 2 * 2 = 512 = shift right 9
const vec_t sum0a = vec_slli_16(vec_max_16(vec_min_16(in0[j*2+0], One), 6), shift);
const vec_t pa = vec_mulhi_16(sum0a, sum1a);  // Fast!
out[j] = vec_packus_16(pa, pb);
```

**nnuebie Approach** (`network.rs:213`):
```rust
let prod = _mm256_mullo_epi16(v0_c, v1_c);  // Slower regular mul
let res = _mm256_srli_epi16(prod, 9);
```

**Expected Impact**: 5-10% improvement

---

### 4. Accumulator Stack - Partial Implementation

**What nnuebie Does** (`accumulator_stack.rs:125-141`):
- `push()` doesn't copy accumulators (good!)
- `update_incremental()` searches backward for computed state
- Copies accumulator data when found

**What Stockfish Does** (`nnue_accumulator.cpp:122-192`):
- **Forward updates**: When a computed state is found, propagates forward to all descendants
- **Backward updates**: When refresh needed, can propagate backward to update ancestors
- More sophisticated lazy evaluation with both forward and backward propagation

**Key insight**: Stockfish's `push()` is truly O(1) - they don't copy anything. Their `pop()` is also O(1). The lazy evaluation happens at evaluation time.

**nnuebie's `update_incremental` is O(n) in search depth** - it copies the entire accumulator every time!

**Expected Impact**: 10-15% improvement in deep search

---

### 5. Missing VNNI for MLP

**Stockfish** (`simd.h:68-77`):
```cpp
#if defined(USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);  // Single instruction!
#else
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    acc = _mm256_add_epi32(acc, product0);
#endif
```

**nnuebie** (`layers.rs:74-88`):
```rust
// Always uses non-VNNI path
let p0 = _mm256_maddubs_epi16(input_vec, w0);
let s0 = _mm256_madd_epi16(p0, ones);
acc0 = _mm256_add_epi32(acc0, s0);
```

---

### 6. No Prefetching

Stockfish uses `_mm_prefetch` for cache optimization in hot paths. nnuebie doesn't use prefetching.

---

## Performance Targets After All Optimizations

| Benchmark | Current | After Phase 2.1-2.3 | Target |
|-----------|---------|---------------------|--------|
| Full Refresh | ~220K/s | ~350K/s | 400K/s |
| Incremental Update | ~1.18M/s | ~2.0M/s | 2.5M/s |

---

## Implementation Plan

### Phase 2.1: Weight Permutation (High Priority)

1. Add permutation function to `feature_transformer.rs`
2. Apply `PackusEpi16Order` pattern at load time
3. Update `transform_features` to work with permuted weights

**Files to modify**:
- `src/feature_transformer.rs` - Add permutation
- `src/network.rs` - Update transform function if needed

### Phase 2.2: MLP Weight Scrambling (High Priority)

1. Add scrambled weight indexing to `AffineTransform`
2. Implement proper VNNI path in `layers.rs`
3. Use `_mm256_dpbusd_epi32` when available

**Files to modify**:
- `src/layers.rs` - Rewrite affine transform

### Phase 2.3: Optimize Feature Transformer Multiply

1. Change from `mullo` + `srli` to `slli` + `mulhi`
2. Ensure weights are scaled by 2 at load time

**Files to modify**:
- `src/network.rs` - Update transform_features_avx2

### Phase 2.4: Improve Accumulator Stack

1. Implement forward propagation when finding computed state
2. Consider backward update optimization

**Files to modify**:
- `src/accumulator_stack.rs`

### Phase 2.5: Add Prefetching

1. Add `_mm_prefetch` calls in hot loops

---

## Phase 1.1: Better Memory Alignment (Memory Alignment)

## Phase 1.2: AccumulatorCaches (Finny Tables)

### What Are Finny Tables?
Stockfish caches partial accumulator updates per king square position. This avoids recalculating from scratch when only the king moves.

### Implementation Details
- Create `AccumulatorCache` struct with 64 entries (one per king square per color)
- Each entry stores the accumulator state for that king position
- On king move: instead of full refresh, copy from cached state + apply piece differences

### Expected Performance Gain
- **2-4x speedup** when king moves (common in chess)
- ~10-20% overall improvement in typical search

### Files to Modify
1. `src/accumulator.rs` - Add `AccumulatorCache` struct
2. `src/nnue.rs` - Integrate cache lookups
3. `src/accumulator_stack.rs` - Update on king moves

### Reference Implementation
See `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_accumulator.h` lines 47-80

---

## Phase 1.3: Weight Permutation

### What Is Weight Permutation?
Stockfish permutes network weights to improve SIMD efficiency. Elements that are processed together in SIMD are stored adjacently in memory.

### Current Implementation
- Current: weights stored in standard file order
- Scale factor: only multiplies by 2

### Stockfish Approach
1. **Reorder weights** - Group features for better cache behavior
2. **Scale by 2** - Already done
3. **VNNI optimization** - Use `_mm256_dpbusd_epi32` for dot products (already implemented)

### Expected Performance Gain
- ~5-10% improvement in MLP evaluation
- Better cache utilization

### Files to Modify
1. `src/feature_transformer.rs` - Add weight permutation in `read_parameters`
2. `src/layers.rs` - Verify VNNI usage

### Reference Implementation
See `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_feature_transformer.h` lines 467-469

---

## Phase 1.4: Accumulator Deep Copy Fix

### Current Problem
In `accumulator_stack.rs`, the `push()` method clones the entire accumulator:
```rust
self.stack[self.current_idx] = self.stack[prev_idx].clone();
```
This copies 24KB (2 x 3072 x 2 bytes) on every move.

### Stockfish Approach
Stockfish uses a **copy-on-write** strategy:
- Stack entries share the same accumulator memory
- Only mark which perspective is "dirty"
- On access, recompute only what's needed

### Expected Performance Gain
- **~50% reduction** in push() overhead
- Significant improvement in make/unmake performance

### Files to Modify
1. `src/accumulator_stack.rs` - Implement copy-on-write
2. `src/accumulator.rs` - Add dirty flag tracking

### Reference Implementation
See `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_accumulator.h` - `State` struct

---

## Phase 1.5: Thread-Local Networks (Optional)

### What Is This?
Stockfish creates a copy of the network for each search thread to avoid lock contention.

### Implementation
```rust
thread_local! {
    static NETWORKS: NnueNetworks = NnueNetworks::new();
}
```

### Expected Performance Gain
- Eliminates mutex contention in multi-threaded search
- ~10-30% improvement with many search threads

### Files to Modify
1. `src/nnue.rs` - Add thread-local storage
2. `src/lib.rs` - Export for thread-local use

---

## Architecture Notes

### Key Stockfish Files for Reference
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_common.h` - Constants (CacheLineSize=64)
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_accumulator.h` - Accumulator + Cache
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_accumulator.cpp` - Implementation (line 122-192 for forward/backward updates)
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_feature_transformer.h` - Feature transformer + permutation (lines 289-319)
- `archive/nnue/Stockfish-sf_17.1/src/nnue/layers/affine_transform.h` - MLP layers with VNNI (lines 153-239)
- `archive/nnue/Stockfish-sf_17.1/src/nnue/layers/simd.h` - SIMD definitions (lines 68-77 for VNNI)
- `archive/nnue/Stockfish-sf_17.1/src/nnue/network.h` - Network structure

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
