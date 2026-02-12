# nnuebie

A high-performance, thread-safe, SIMD-accelerated NNUE (Efficiently Updatable Neural Network) inference library for Chess engines written in Rust.

This crate provides a drop-in evaluation component compatible with Stockfish-trained NNUE networks (SFNNv9 architecture). It is designed for maximum throughput, supporting both single-threaded engines and highly parallel environments (LazySMP) with minimal memory overhead.

## Features

-   **High Performance**: Uses AVX2 SIMD intrinsics for both accumulator updates and the forward pass.
-   **Thread Safety**: Separates read-only network weights (`Arc<NnueNetworks>`) from thread-local state (`NNUEProbe`), allowing thousands of threads to share a single copy of the heavy network data.
-   **Incremental Updates**: Efficiently updates accumulators based on moves (added/removed pieces) rather than refreshing from scratch.
-   **Stockfish Compatibility**: Loads standard `.nnue` files (Big/Small networks) used by modern Stockfish versions. Make sure the Hashes Match. This crate strictly only supports `HalfKa_hm_v2`

## Usage

```bash
# Enable AVX2/Native optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Single-Threaded Example

```rust
use nnuebie::{NNUEProbe, Piece, Color};

fn main() -> std::io::Result<()> {
    let mut probe = NNUEProbe::new("big.nnue", "small.nnue")?;

    // Set up board (simplified example)
    let pieces = vec![(Piece::WhiteKing, 4), (Piece::BlackKing, 60)];
    probe.set_position(&pieces);

    let score = probe.evaluate(Color::White);
    println!("Evaluation: {}", score);
    
    Ok(())
}
```

### Multi-Threaded Example (LazySMP)

For parallel search, load the networks once and share them.

```rust
use nnuebie::{NNUEProbe, NnueNetworks, Piece, Color};
use std::sync::Arc;
use std::thread;

fn main() {
    // 1. Load networks (Heavy I/O)
    let networks = Arc::new(NnueNetworks::new("big.nnue", "small.nnue").unwrap());

    // 2. Spawn threads
    let mut handles = vec![];
    for _ in 0..8 {
        let net_ref = networks.clone();
        handles.push(thread::spawn(move || {
            // 3. Create lightweight thread-local probe
            let mut probe = NNUEProbe::with_networks(net_ref).unwrap();
            
            // Use probe...
            // probe.set_position(...)
            // probe.update(...)
            // probe.evaluate(...)
        }));
    }
    
    for h in handles { h.join().unwrap(); }
}
```

## Performance

Benchmarks on an 8-core machine with `target-cpu=native`:

| Operation | Throughput (Total) | Throughput (Per Thread) |
|-----------|--------------------|-------------------------|
| **Full Refresh** (8 threads) | ~1,160,000 evals/sec | ~145,000 evals/sec |
| **Incremental** (8 threads) | ~7,500,000 evals/sec | ~938,000 evals/sec |

*Benchmarks run using `cargo run --release --bin benchmark_multithread`.*
