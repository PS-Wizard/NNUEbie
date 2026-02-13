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

    // Position with king on e1
    let pieces = vec![
        (Piece::WhiteKing, 4),  // e1
        (Piece::WhitePawn, 8),  // a2
        (Piece::WhitePawn, 9),  // b2
        (Piece::WhitePawn, 10), // c2
        (Piece::WhitePawn, 11), // d2
        (Piece::WhitePawn, 12), // e2
        (Piece::WhitePawn, 13), // f2
        (Piece::WhitePawn, 14), // g2
        (Piece::WhitePawn, 15), // h2
        (Piece::BlackKing, 60), // e8
        (Piece::BlackPawn, 48), // a7
        (Piece::BlackPawn, 49), // b7
        (Piece::BlackPawn, 50), // c7
        (Piece::BlackPawn, 51), // d7
        (Piece::BlackPawn, 52), // e7
        (Piece::BlackPawn, 53), // f7
        (Piece::BlackPawn, 54), // g7
        (Piece::BlackPawn, 55), // h7
    ];

    probe.set_position(&pieces);

    let iterations = 1_000_000;

    println!("\n=== Testing King Move Performance ===\n");

    // Test 1: King moves (e1->e2->e1) with Finny Tables
    println!("Test 1: King moves using Finny Tables (with cache)");
    let start = Instant::now();
    for i in 0..iterations {
        // King e1 -> e2
        probe.make_move(4, 12, Piece::WhiteKing);
        std::hint::black_box(probe.evaluate(Color::White));

        // King e2 -> e1 (back) - should hit cache!
        probe.make_move(12, 4, Piece::WhiteKing);
        std::hint::black_box(probe.evaluate(Color::White));

        // Unmake moves to return to root (O(1) each!)
        probe.unmake_move(12, 4, Piece::WhiteKing, None);
        probe.unmake_move(4, 12, Piece::WhiteKing, None);

        if i % 100000 == 0 && i > 0 {
            print!(".");
        }
    }
    let finny_duration = start.elapsed();
    let finny_nps = (iterations * 2) as f64 / finny_duration.as_secs_f64();
    println!("\nWith Finny Tables: {:.2} king moves/sec", finny_nps);
    println!(
        "Time per king move cycle: {:.3} µs",
        finny_duration.as_secs_f64() * 1_000_000.0 / (iterations as f64 * 2.0)
    );

    // Reset and test without cache (simulated)
    println!("\nTest 2: King moves without cache (simulated full refresh)");
    probe.set_position(&pieces);

    let start = Instant::now();
    for i in 0..iterations {
        // Simulate: set position from scratch each time (worst case)
        probe.set_position(&pieces);

        // Move king to e2
        let pieces_e2 = vec![
            (Piece::WhiteKing, 12), // e2
            (Piece::WhitePawn, 8),
            (Piece::WhitePawn, 9),
            (Piece::WhitePawn, 10),
            (Piece::WhitePawn, 11),
            (Piece::WhitePawn, 13),
            (Piece::WhitePawn, 14),
            (Piece::WhitePawn, 15),
            (Piece::BlackKing, 60),
            (Piece::BlackPawn, 48),
            (Piece::BlackPawn, 49),
            (Piece::BlackPawn, 50),
            (Piece::BlackPawn, 51),
            (Piece::BlackPawn, 52),
            (Piece::BlackPawn, 53),
            (Piece::BlackPawn, 54),
            (Piece::BlackPawn, 55),
        ];
        probe.set_position(&pieces_e2);
        std::hint::black_box(probe.evaluate(Color::White));

        // Move king back to e1
        probe.set_position(&pieces);
        std::hint::black_box(probe.evaluate(Color::White));

        if i % 100000 == 0 && i > 0 {
            print!(".");
        }
    }
    let full_duration = start.elapsed();
    let full_nps = (iterations * 2) as f64 / full_duration.as_secs_f64();
    println!("\nWith full refresh: {:.2} king moves/sec", full_nps);
    println!(
        "Time per king move cycle: {:.3} µs",
        full_duration.as_secs_f64() * 1_000_000.0 / (iterations as f64 * 2.0)
    );

    // Summary
    println!("\n=== Summary ===");
    println!(
        "Finny Tables speedup: {:.1}x faster than full refresh",
        finny_nps / full_nps
    );
    println!("\nIn a chess engine with active king (endgame):");
    println!("  - Without Finny Tables: king moves cost full refresh");
    println!("  - With Finny Tables: returning king to cached position is instant");
}
