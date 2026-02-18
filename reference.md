# nnuebie Technical Reference & Roadmap

## 🚀 Current Implementation Status

**nnuebie** currently achieves full architectural feature parity with Stockfish 17.1 NNUE. It implements the SFNNv9 architecture (HalfKA_hm_v2) with high-performance incremental updates.

### ✅ Implemented Features

#### Core Architecture
- **HalfKA Structure:** Supports the standard 768->3072->1 (Small) and Big network architectures.
- **Lazy Evaluation:** Uses the small network for preliminary evaluation, falling back to the big network only when complexity warrants it.
- **Efficient Feature Transformer:**
    - Weights permutation handling matches Stockfish.
    - Correct input scaling (x2) and output scaling.

#### Accumulator Management (The "Finny Tables")
- **Accumulator Cache (Finny Tables):**
    - Maintains a per-thread cache of `Accumulator` states + `Bitboards` for all 64 King squares.
    - **Impact:** King moves are now **O(1)** operations (updating cache incrementally) instead of O(N) full refreshes.
- **Tiled Cache Updates:**
    - Uses AVX2 tiled kernels (`refresh_avx2_3072`) to update cache entries.
    - Updates cache and target accumulator in a single pass to minimize memory traffic.
- **Accumulator Stack:**
    - Supports `AccumulatorStack` with push/pop/reset.
    - **Backward Propagation:** Can update "backwards" from a future computed state (e.g., in search) to fill gaps.
    - **Correct King Move Detection:** Automatically detects king moves and transparently switches between cache-based refresh and standard incremental updates.

#### SIMD Optimizations (AVX2)
- **Accumulator Operations:**
    - `refresh_avx2_3072`: Tiled refresh for big network (12 tiles of 256).
    - `refresh_avx2_128`: Single-tile refresh for small network.
    - `update_and_copy_avx2`: Optimized kernel for updating Finny Tables.
- **Linear Layers (AffineTransform):**
    - `propagate_avx2`: Optimized 8-bit dot product using `maddubs` + `madd`.
    - **Sparse Input Optimization:** Skips processing 32-byte chunks of input if they are all zero (crucial for ReLU outputs).
- **Activations:**
    - `ClippedReLU`: Branchless AVX2 implementation using `packus`.
    - `SqrClippedReLU`: Vectorized square-and-scale implementation.

#### Memory
- **Aligned Allocation:** All heavy buffers (weights, accumulators) are heap-allocated with 64-byte alignment to prevent cache line splits and enable SIMD loads.

---

## 🔮 Roadmap: Performance & Features

While the current implementation matches Stockfish's logic, modern hardware offers instructions that can push performance further.

### 1. AVX-512 & VNNI Support (High Priority)
The current AVX2 implementation simulates 8-bit integer dot products using `_mm256_maddubs_epi16` (multiply add unsigned byte signed) followed by `_mm256_madd_epi16`.

-   **VNNI (`vpdpbusd`):** AVX-512 VNNI (and AVX-VNNI) provides a single instruction to do `INT8` dot products accumulation into `INT32`. This effectively doubles throughput for the dense layers.
-   **512-bit Registers:** Using ZMM registers (512-bit) for accumulator updates would double the throughput of the `refresh` and `update` kernels (processing 32 `i16`s at once instead of 16).

### 3. Micro-Optimizations
-   **Prefetching:** Explicit `_mm_prefetch` for weights in the accumulator update loops to hide memory latency.
-   **Huge Pages:** Allocating the large network weights (hundreds of MBs) using Huge Pages to reduce TLB misses.
-   **Input buffer alignment:** Ensure all `DirtyPiece` lists and scratch buffers are cache-line aligned.

### 4. Code Structure
-   **Kernel Dispatch:** Improve the CPU feature detection to select the best kernel at runtime with less overhead (currently checks `is_x86_feature_detected` inside wrapper functions or relies on compile-time flags).
