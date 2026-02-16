use nnuebie::{Color, NNUEProbe, NnueNetworks, Piece};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

fn main() {
    let big_path = "archive/nnue/networks/nn-1c0000000000.nnue";
    let small_path = "archive/nnue/networks/nn-37f18f62d772.nnue";

    println!("Loading networks...");

    // Use fallback if running from target/release directory
    let networks = NnueNetworks::new(big_path, small_path)
        .or_else(|_| {
            NnueNetworks::new(
                "../../archive/nnue/networks/nn-1c0000000000.nnue",
                "../../archive/nnue/networks/nn-37f18f62d772.nnue",
            )
        })
        .expect("Failed to load networks");

    let networks = Arc::new(networks);

    let num_threads = 8;
    let iterations_per_thread = 200_000; // Total 1.6M evals

    println!(
        "Benchmarking with {} threads, {} iterations each (Total {} evals)...",
        num_threads,
        iterations_per_thread,
        num_threads * iterations_per_thread
    );

    // Position: Startpos
    let pieces = vec![
        (Piece::WhiteRook, 0),
        (Piece::WhiteKnight, 1),
        (Piece::WhiteBishop, 2),
        (Piece::WhiteQueen, 3),
        (Piece::WhiteKing, 4),
        (Piece::WhiteBishop, 5),
        (Piece::WhiteKnight, 6),
        (Piece::WhiteRook, 7),
        (Piece::WhitePawn, 8),
        (Piece::WhitePawn, 9),
        (Piece::WhitePawn, 10),
        (Piece::WhitePawn, 11),
        (Piece::WhitePawn, 12),
        (Piece::WhitePawn, 13),
        (Piece::WhitePawn, 14),
        (Piece::WhitePawn, 15),
        (Piece::BlackPawn, 48),
        (Piece::BlackPawn, 49),
        (Piece::BlackPawn, 50),
        (Piece::BlackPawn, 51),
        (Piece::BlackPawn, 52),
        (Piece::BlackPawn, 53),
        (Piece::BlackPawn, 54),
        (Piece::BlackPawn, 55),
        (Piece::BlackRook, 56),
        (Piece::BlackKnight, 57),
        (Piece::BlackBishop, 58),
        (Piece::BlackQueen, 59),
        (Piece::BlackKing, 60),
        (Piece::BlackBishop, 61),
        (Piece::BlackKnight, 62),
        (Piece::BlackRook, 63),
    ];

    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let mut handles = vec![];

    // Spawn threads
    for _t_id in 0..num_threads {
        let networks_clone = networks.clone();
        let pieces_clone = pieces.clone();
        let barrier_clone = barrier.clone();

        let handle = thread::spawn(move || {
            let mut probe =
                NNUEProbe::with_networks(networks_clone).expect("Failed to create probe");

            // Warmup
            probe.set_position(&pieces_clone, 0);
            for _ in 0..100 {
                std::hint::black_box(probe.evaluate(Color::White));
            }

            // Wait for all threads to be ready
            barrier_clone.wait();

            // Benchmark loop (Full Refresh)
            for _ in 0..iterations_per_thread {
                // Simulate setting a new position (invalidates accumulators -> triggers refresh)
                probe.set_position(&pieces_clone, 0);
                std::hint::black_box(probe.evaluate(Color::White));
            }

            // Wait for all threads to finish refresh benchmark
            barrier_clone.wait();

            // Benchmark Incremental Update
            let iterations_inc = 1_000_000;
            probe.set_position(&pieces_clone, 0);

            // Sync before starting incremental phase
            barrier_clone.wait();

            for i in 0..iterations_inc {
                if i % 2 == 0 {
                    probe.update(&[(Piece::WhitePawn, 12)], &[(Piece::WhitePawn, 28)]);
                } else {
                    probe.update(&[(Piece::WhitePawn, 28)], &[(Piece::WhitePawn, 12)]);
                }
                std::hint::black_box(probe.evaluate(Color::White));
            }
        });
        handles.push(handle);
    }

    println!("Waiting for threads to initialize...");
    // Wait for threads to reach the starting line
    barrier.wait();
    let start_refresh = Instant::now();

    // Wait for refresh benchmark to finish (barrier index 1)
    // Note: The main thread needs to wait on the barrier too if we want to time it accurately from main.
    // However, the barrier is initialized with `num_threads + 1`.
    // So main thread must call wait() for each synchronization point.

    // 1. Start Refresh
    // (Already called barrier.wait() above)

    // 2. End Refresh / Start Incremental Prep
    barrier.wait();
    let duration_refresh = start_refresh.elapsed();

    // 3. Start Incremental
    barrier.wait();
    let start_inc = Instant::now();

    for handle in handles {
        handle.join().unwrap();
    }
    let duration_inc = start_inc.elapsed();

    // Calculate Refresh Stats
    let total_evals_refresh = num_threads * iterations_per_thread;
    let nps_refresh = total_evals_refresh as f64 / duration_refresh.as_secs_f64();

    println!("--------------------------------------------------");
    println!("Full Refresh ({} threads)", num_threads);
    println!("Total evaluations: {}", total_evals_refresh);
    println!("Time taken:        {:.4}s", duration_refresh.as_secs_f64());
    println!("Total Throughput:  {:.2} evals/sec", nps_refresh);
    println!(
        "Per-thread Speed:  {:.2} evals/sec",
        nps_refresh / num_threads as f64
    );
    println!("--------------------------------------------------");

    // Calculate Incremental Stats
    let iterations_inc_per_thread = 1_000_000;
    let total_evals_inc = num_threads * iterations_inc_per_thread;
    let nps_inc = total_evals_inc as f64 / duration_inc.as_secs_f64();

    println!("Incremental Update ({} threads)", num_threads);
    println!("Total evaluations: {}", total_evals_inc);
    println!("Time taken:        {:.4}s", duration_inc.as_secs_f64());
    println!("Total Throughput:  {:.2} evals/sec", nps_inc);
    println!(
        "Per-thread Speed:  {:.2} evals/sec",
        nps_inc / num_threads as f64
    );
    println!("--------------------------------------------------");
}
