use nnuebie::{Color, NNUEProbe, Piece};
use std::time::Instant;

fn main() {
    let big_path = "archive/nnue/networks/nn-1c0000000000.nnue";
    let small_path = "archive/nnue/networks/nn-37f18f62d772.nnue";

    println!("Loading networks...");

    let mut probe = match NNUEProbe::new(big_path, small_path) {
        Ok(p) => p,
        Err(_) => NNUEProbe::new(
            "../../archive/nnue/networks/nn-1c0000000000.nnue",
            "../../archive/nnue/networks/nn-37f18f62d772.nnue",
        )
        .expect("Failed to load networks"),
    };

    // Position: Startpos
    // rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
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

    // Set initial position
    probe.set_position(&pieces);

    // Warmup
    for _ in 0..100 {
        std::hint::black_box(probe.evaluate(Color::White));
    }

    let iterations = 1_000_000;
    println!("Benchmarking {} evaluations...", iterations);

    // Benchmark Full Refresh
    let start = Instant::now();
    for _ in 0..iterations {
        probe.set_position(&pieces);
        std::hint::black_box(probe.evaluate(Color::White));
    }
    let duration = start.elapsed();
    let nps = iterations as f64 / duration.as_secs_f64();
    println!("Full Refresh Throughput: {:.2} evaluations/sec", nps);

    // Benchmark Incremental Update
    println!("\nBenchmarking Incremental Update...");

    // Reset to initial position
    probe.set_position(&pieces);

    let iterations_inc = 1_000_000;

    let start = Instant::now();
    for i in 0..iterations_inc {
        if i % 2 == 0 {
            // Move pawn e2 -> e4
            probe.update(&[(Piece::WhitePawn, 12)], &[(Piece::WhitePawn, 28)]);
        } else {
            // Move pawn e4 -> e2 (undo)
            probe.update(&[(Piece::WhitePawn, 28)], &[(Piece::WhitePawn, 12)]);
        }
        std::hint::black_box(probe.evaluate(Color::White));
    }
    let duration = start.elapsed();
    let nps_inc = iterations_inc as f64 / duration.as_secs_f64();
    println!(
        "Incremental Update Throughput: {:.2} evaluations/sec",
        nps_inc
    );
    println!("Speedup: {:.2}x", nps_inc / nps);
}
