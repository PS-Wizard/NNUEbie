# Before SIMD Accumulator

[wizard@archbtw ~/Projects/nnue-rs/target/release] -> time ./benchmark
Loading networks...
Benchmarking 1000000 evaluations...
Full Refresh Throughput: 11262.40 evaluations/sec

Benchmarking Incremental Update...
Incremental Update Throughput: 79992.62 evaluations/sec
Speedup: 7.10x
./benchmark  101.17s user 0.04s system 99% cpu 1:41.47 total
[wizard@archbtw ~/Projects/nnue-rs/target/release] ->


# After SIMD Accumulator
[wizard@archbtw ~/Projects/nnue-rs/target/release] -> time ./benchmark
Loading networks...
Benchmarking 1000000 evaluations...
Full Refresh Throughput: 110601.51 evaluations/sec
Benchmarking Incremental Update...
Incremental Update Throughput: 245491.64 evaluations/sec
Speedup: 2.22x
./benchmark  13.19s user 0.03s system 99% cpu 13.261 total
[wizard@archbtw ~/Projects/nnue-rs/target/release] ->


# After SIMD Forward Pass

[wizard@archbtw ~/Projects/nnue-rs/target/release] -> time ./benchmark
Loading networks...
Benchmarking 1000000 evaluations...
Full Refresh Throughput: 165569.23 evaluations/sec

Benchmarking Incremental Update...
Incremental Update Throughput: 1045634.17 evaluations/sec
Speedup: 6.32x
./benchmark  7.08s user 0.03s system 99% cpu 7.139 total
[wizard@archbtw ~/Projects/nnue-rs/target/release] ->
