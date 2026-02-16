# NNUE Performance Optimization Roadmap

## Overview

This document outlines the performance optimization phases for the nnuebie Rust implementation of Stockfish's NNUE neural network evaluation.

## Current Status

- **Phase 1.1**: Memory Alignment - DONE
- **Phase 1.2**: Finny Tables (AccumulatorCaches) - DONE
- **Phase 1.3**: Weight Permutation - **CRITICAL - NOT IMPLEMENTED**
- **Phase 1.4**: Accumulator Deep Copy Fix - **PARTIALLY IMPLEMENTED**

## Current Performance

| Benchmark | Current | Target |
|-----------|---------|--------|
| Full Refresh | ~220K/s | 400K/s |
| Incremental Update | ~1.18M/s | 2.5M/s |

---

## Phase 1.3 Deep Dive: Weight Permutation

### What is Weight Permutation?

Weight permutation is a memory layout optimization Stockfish applies at **load time** to reorganize network weights for optimal SIMD behavior. It's specifically about the `_mm256_packus_epi16` instruction used in the feature transformer.

**The Problem**: `_mm256_packus_epi16` takes two 256-bit vectors (containing sixteen 16-bit integers each) and interleaves them into one 256-bit vector of 8-bit results with a **specific ordering**:
- First: elements 0,2,4,6,... from both input vectors
- Then: elements 1,3,5,7,... from both input vectors

If weights are stored in standard row-major order, the pack instruction produces **wrong results**. Stockfish solves this by permuting weights at load time so the pack instruction produces correct output directly.

### Stockfish's Implementation (`nnue_feature_transformer.h:264-319`)

**Permutation Order for AVX2:**
```cpp
static constexpr auto PackusEpi16Order = []() -> std::array<std::size_t, 8> {
    return {0, 2, 1, 3, 4, 6, 5, 7};  // AVX2
}();
```

**The permutation function (lines 158-185):**
```cpp
template<std::size_t BlockSize, typename T, std::size_t N, std::size_t OrderSize>
void permute(T (&data)[N], const std::array<std::size_t, OrderSize>& order) {
    // Divides data into blocks of BlockSize * OrderSize bytes
    // Reorders each group according to 'order' array
    // BlockSize=16 (i16 = 2 bytes, so 16 i16s = 32 bytes = one 256-bit vector)
}
```

**In read_parameters (lines 312-321):**
```cpp
bool read_parameters(std::istream& stream) {
    read_leb_128<BiasType>(stream, biases, HalfDimensions);
    read_leb_128<WeightType>(stream, weights, ...);
    read_leb_128<PSQTWeightType>(stream, psqtWeights, ...);
    
    permute_weights();      // <-- MISSING IN NNU EBIE
    scale_weights(true);   // Scale by 2 after permutation
    return !stream.fail();
}
```

### What nnuebie Does vs. What It Should Do

**Current (`feature_transformer.rs:26-48`):**
```rust
pub fn read_parameters(&mut self, reader: &mut R, skip_first_magic: bool) -> io::Result<()> {
    // ... read weights
    // Scale by 2 only - NO PERMUTATION!
    for b in &mut biases_vec { *b = b.wrapping_mul(2); }
    for w in &mut weights_vec { *w = w.wrapping_mul(2); }
    // Convert to AlignedBuffer
    ...
}
```

**What's Missing:**
1. Apply permutation after reading, before scaling
2. The permutation must happen in 16-element blocks (256-bit = 16 x i16)

### How to Implement in nnuebie

```rust
// In feature_transformer.rs

// AVX2 permutation order
const PACKUS_EPI16_ORDER: [usize; 8] = [0, 2, 1, 3, 4, 6, 5, 7];

fn permute_weights(data: &mut [i16]) {
    const BLOCK_SIZE: usize = 16; // 16 i16 = 32 bytes = 256 bits
    const ORDER_SIZE: usize = 8;
    const CHUNK_SIZE: usize = BLOCK_SIZE * ORDER_SIZE; // 128 elements per chunk
    
    let mut buffer = vec![0i16; CHUNK_SIZE];
    
    let mut i = 0;
    while i < data.len() {
        // Process one chunk at a time
        for j in 0..ORDER_SIZE {
            let src_offset = i + PACKUS_EPI16_ORDER[j] * BLOCK_SIZE;
            buffer[j * BLOCK_SIZE..(j + 1) * BLOCK_SIZE]
                .copy_from_slice(&data[src_offset..src_offset + BLOCK_SIZE]);
        }
        data[i..i + CHUNK_SIZE].copy_from_slice(&buffer);
        i += CHUNK_SIZE;
    }
}

pub fn read_parameters(&mut self, reader: &mut R, skip_first_magic: bool) -> io::Result<()> {
    // ... read weights
    permute_weights(&mut biases_vec);  // ADD THIS
    permute_weights(&mut weights_vec); // ADD THIS
    // Then scale by 2
    for b in &mut biases_vec { *b = b.wrapping_mul(2); }
    for w in &mut weights_vec { *w = w.wrapping_mul(2); }
    // ...
}
```

---

## Phase 1.4 Deep Dive: Accumulator Deep Copy Fix

### Current State: Partially Implemented

**What's Done:**
- `push()` does NOT clone accumulators (good!)
- Uses lazy evaluation - finds last computed state

**What's Still Wrong:**
In `accumulator_stack.rs:177-210`, `update_incremental()` still **copies entire accumulator** when it finds a computed state:

```rust
let source_accum: Option<(...)> = if self.stack[last_computed_idx].computed == [true, true] {
    let source = &self.stack[last_computed_idx];
    Some((
        source.acc_big.accumulation[0].as_slice().to_vec(),  // COPY!
        source.acc_big.accumulation[1].as_slice().to_vec(),  // COPY!
        // ...
    ))
} else { None };
```

This copies ~12KB per update - not acceptable for O(1) push.

### What Stockfish Does (Truly O(1))

Stockfish's `push()` does literally **nothing** - no copying at all:
- Only marks state as "dirty"
- At evaluation time, lazily finds last computed state and updates from there

The key insight: Stockfish doesn't copy in update_incremental. Instead:
1. They store a reference/pointer to the previous computed state
2. At evaluation time, they propagate forward from that state

### Fix for nnuebie

Remove the copy in `update_incremental()`. Instead, store a pointer/index to the source state and use pointer arithmetic for updates. The key is that the actual copying happens implicitly through shared memory references.

---

## Why VNNI Gives Incorrect Results

VNNI requires **two** things to work correctly:

### 1. Weight Scrambling in AffineTransform

Stockfish (`affine_transform.h:153-156`):
```cpp
static constexpr IndexType get_weight_index_scrambled(IndexType i) {
    return (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4
         + i / PaddedInputDimensions * 4 + i % 4;
}
```

This reorders weights so VNNI's dot product instruction (`_mm256_dpbusd_epi32`) works correctly. The pattern transposes the weight matrix in 4x4 blocks.

### 2. Proper Feature Transformer Permutation

The weights must be permuted as described above for `packus` to work.

### What's Wrong in nnuebie:

1. **No weight scrambling in layers.rs** - Uses row-major order (`j * padded_input_dims + i`)
2. **No permutation in feature_transformer.rs** - Weights aren't reordered for packus
3. **No VNNI path in layers.rs** - Uses `_mm256_maddubs_epi16` + `_mm256_madd_epi16` instead of `_mm256_dpbusd_epi32`

The current implementation in `layers.rs:74-88`:
```rust
// maddubs: input unsigned, weight signed
let p0 = _mm256_maddubs_epi16(input_vec, w0);
let s0 = _mm256_madd_epi16(p0, ones);  // Extra instruction
acc0 = _mm256_add_epi32(acc0, s0);
```

Stockfish uses VNNI when available (simd.h:68-77):
```cpp
#if defined(USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);  // Single instruction!
#else
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    acc = _mm256_add_epi32(acc, product0);
#endif
```

### Summary: Why Your VNNI Attempts Failed

When you tried VNNI, you likely got wrong evaluations because:
1. Without weight permutation, the packus instruction produces scrambled output
2. Without weight scrambling, the VNNI dot product computes wrong values (wrong weight × input pairings)
3. These errors compound through the network layers

The validation should fail if you compare against Stockfish's output.

---

## Critical Issues Summary

### 1. Weight Permutation NOT Implemented (Phase 1.3 - CRITICAL)

**Problem**: nnuebie reads weights in standard file order, but Stockfish permutes weights at load time for optimal SIMD behavior.

**Expected Impact**: 15-30% improvement in feature transformation

### 2. MLP Weight Scrambling NOT Implemented (CRITICAL)

**Problem**: Stockfish uses scrambled weight ordering in affine transforms to enable VNNI dot product instructions.

**Expected Impact**: 10-20% improvement in MLP propagation

### 3. Feature Transformer Multiplication Not Optimized

**Problem**: Stockfish uses `mulhi` for faster multiplication. -- DONE in nnuebie

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

### 4. Missing VNNI for MLP

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

### 5. No Prefetching

Stockfish uses `_mm_prefetch` for cache optimization in hot paths. nnuebie doesn't use prefetching.

---

## Implementation Plan

### Phase 2.1: Weight Permutation (High Priority)

1. Add permutation function to `feature_transformer.rs`
2. Apply `PackusEpi16Order` pattern at load time
3. Update `transform_features` to work with permuted weights

**Files to modify**:
- `src/feature_transformer.rs` - Add permutation

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

1. Remove accumulator copy in `update_incremental()`
2. Use reference/index instead of copying

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
3. **VNNI optimization** - Use `_mm256_dpbusd_epi32` for dot products

### Expected Performance Gain
- ~5-10% improvement in MLP evaluation
- Better cache utilization

### Files to Modify
1. `src/feature_transformer.rs` - Add weight permutation in `read_parameters`
2. `src/layers.rs` - Implement weight scrambling for VNNI

### Reference Implementation
See `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_feature_transformer.h` lines 289-319

---

## Phase 1.4: Accumulator Deep Copy Fix

### Current Problem
In `accumulator_stack.rs`, the `update_incremental()` method copies entire accumulators:
```rust
source.acc_big.accumulation[0].as_slice().to_vec(),  // COPY!
```
This copies ~12KB every time a computed state is found.

### Stockfish Approach
Stockfish uses a **copy-on-write** strategy:
- Stack entries share the same accumulator memory
- Only mark which perspective is "dirty"
- On access, recompute only what's needed

### Expected Performance Gain
- **~50% reduction** in push() overhead
- Significant improvement in make/unmake performance

### Files to Modify
1. `src/accumulator_stack.rs` - Remove copying in update_incremental

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
