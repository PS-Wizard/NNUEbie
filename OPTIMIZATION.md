# NNUE Implementation Roadmap

## Overview
This document outlines the improvements needed to make nnuebie a production-quality NNUE evaluation library, based on comparison with Stockfish 17.1.

---

## 🚨 Critical Issues (Priority 1)

### 1. King Move Detection in AccumulatorStack
**Current State**: Broken/Incomplete
- **Problem**: Our `update_incremental` chains blindly through states even when a king moves
- **Stockfish**: Has `find_last_usable_accumulator` that stops at any state requiring refresh
- **Action**: Implement proper king move detection per perspective, similar to Stockfish's `find_last_usable_accumulator`

```cpp
// Stockfish implementation
template<Color Perspective, ...>
std::size_t find_last_usable_accumulator() const noexcept {
    for (curr_idx from m_current_idx-1 down to 1) {
        if (accumulator[perspective].computed) return curr_idx;
        if (FeatureSet::requires_refresh(dirtyPiece, Perspective)) return curr_idx;
    }
    return 0;
}
```

### 2. In-Place Cache Updates (Finny Tables)
**Current State**: Partial
- **Problem**: We copy from cache to accumulator, then apply changes. Stockfish updates cache entry in-place first, then memcpy.
- **Action**: Wire up `refresh_in_place` properly or use Stockfish's approach in `update_accumulator_refresh_cache`

---

## ⚡ Performance Optimizations (Priority 2)

### 3. VNNI Support for Affine Layers
**Current State**: AVX2 only (suboptimal)
- **Problem**: We use `_mm256_maddubs_epi16` + `_mm256_madd_epi16` 
- **Stockfish**: Uses `_mm512_dpbusd_epi32` (VNNI) or `_mm256_dpbusd_epi32` (AVX-VNNI) when available
- **Action**: Add VNNI kernel using `vpdpbusd` intrinsic

### 4. Backward Update for Remarginalization
**Current State**: Missing
- **Problem**: We only have forward propagation
- **Stockfish**: Has `backward_update_incremental` for efficient remarginalization
- **Action**: Implement backward update to reuse computed states when possible

### 5. Sparse Input Optimization
**Current State**: Missing
- **Problem**: We process all inputs even when most are zero
- **Stockfish**: Only processes non-zero input indices in affine layers
- **Action**: Add sparse input path that skips zero inputs

---

## 🏗️ Architecture Improvements (Priority 3)

### 6. Remove Hardcoded Dimensions
**Current State**: Hardcoded values everywhere
```rust
// Current (bad)
let (input_dims, half_dims, l2, l3) = if is_big {
    (22528, 3072, 15, 32)
} else {
    (22528, 128, 15, 32)
};
```
- **Stockfish**: Uses template parameters in headers
- **Action**: Create proper network configuration structs

### 7. Centralize Constants
**Current State**: Scattered
- **Action**: Create `nnue_constants.rs` with all architecture constants

### 8. Proper Type System
**Current State**: Inconsistent representations
- **Action**: Standardize on using enums (`Color`, `Piece`) consistently throughout

---

## 🧪 Missing Features (Priority 4)

### 9. Thread-Local Caches
**Current State**: Single global cache
- **Stockfish**: Per-thread accumulator caches
- **Action**: Implement thread-local Finny Tables

### 10. Network Validation
**Current State**: Basic
- **Action**: Add more comprehensive validation against Stockfish

### 11. Weight Permutation Verification
**Current State**: Unknown if correct
- **Action**: Verify that our permutation matches Stockfish's `get_weight_index_scrambled`

---

## ✅ Completed (Reference)

- Phase 1.1: Memory Alignment (AlignedBuffer)
- Phase 1.2: Consolidate types
- Phase 1.3: Remove duplicate evaluation path
- Phase 1.4: Fix hot path Vec allocations
- Phase 1.5: Centralize constants
- Phase 1.6: Clean up benchmark binaries

---

## Architecture Comparison

| Feature | Stockfish | nnuebie (Current) |
|---------|-----------|-------------------|
| **King Move Detection** | `find_last_usable_accumulator` | Partial |
| **Cache Updates** | In-place update + memcpy | Copy-then-apply |
| **VNNI** | Native `vpdpbusd` | AVX2 simulation |
| **Backward Update** | Implemented | Missing |
| **Sparse Inputs** | Optimized | Not implemented |
| **Thread Caches** | Per-thread | Global only |
| **Network Config** | Template parameters | Hardcoded |

---

## Implementation Priority Order

1. Fix king move detection (`find_last_usable_accumulator`)
2. Add VNNI support to affine layers  
3. Implement backward update incremental
4. Add sparse input optimization
5. Thread-local caches (if needed for multithreading)
6. Clean up hardcoded dimensions
