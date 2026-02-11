# Before

[wizard@archbtw ~/Projects/nnue-rs/target/release] -> time ./benchmark
Loading networks...
Benchmarking 1000000 evaluations...
Full Refresh Throughput: 11262.40 evaluations/sec

Benchmarking Incremental Update...
Incremental Update Throughput: 79992.62 evaluations/sec
Speedup: 7.10x
./benchmark  101.17s user 0.04s system 99% cpu 1:41.47 total
[wizard@archbtw ~/Projects/nnue-rs/target/release] ->


# After:
[wizard@archbtw ~/Projects/nnue-rs/target/release] -> time ./benchmark
Loading networks...
Benchmarking 1000000 evaluations...
Full Refresh Throughput: 116830.91 evaluations/sec

Benchmarking Incremental Update...
Incremental Update Throughput: 3643729.68 evaluations/sec
Speedup: 31.19x
./benchmark  8.92s user 0.03s system 99% cpu 8.981 total
[wizard@archbtw ~/Projects/nnue-rs/target/release] ->
