# NNUE Performance Optimization Roadmap

## Overview

This document outlines the performance optimization phases for the nnuebie Rust implementation of Stockfish's NNUE neural network evaluation.

## Current Status

- **Phase 1.1**: Memory Alignment - Pending 
- **Phase 1.2**: Finny Tables (AccumulatorCaches) - Pending
- **Phase 1.3**: Weight Permutation - Pending
- **Phase 1.4**: Accumulator Deep Copy Fix - Pending


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

## Performance Targets

| Benchmark | Current | Target | Phase |
|-----------|---------|--------|-------|
| Full Refresh | ~170K/s | 200K/s | 1.3 |
| Incremental (non-king) | ~2.3M/s | 3M/s | 1.2 |
| Incremental (king move) | ~2.3M/s | 5M/s | 1.2 |
| Make/Unmake | ~2.3M/s | 4M/s | 1.4 |

---

## Architecture Notes

### Key Stockfish Files for Reference
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_common.h` - Constants (CacheLineSize=64)
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_accumulator.h` - Accumulator + Cache
- `archive/nnue/Stockfish-sf_17.1/src/nnue/nnue_feature_transformer.h` - Feature transformer
- `archive/nnue/Stockfish-sf_17.1/src/nnue/layers/affine_transform.h` - MLP layers with VNNI
- `archive/nnue/Stockfish-sf_17.1/src/nnue/network.h` - Network structure

### Memory Alignment Requirements
- All SIMD data: 64-byte aligned
- Use `_mm256_load_si256` (not `_mm256_loadu_si256`) for aligned memory
- Cache line size: 64 bytes

### SIMD Instructions Used
- AVX2: `_mm256_dpbusd_epi32` (VNNI dot product)
- AVX2: `_mm256_maddubs_epi16` (byte * signed byte -> word)
- AVX2: `_mm256_madd_epi16` (word * word -> dword)
- AVX512: `_mm512_dpbusd_epi32` (512-bit VNNI)
