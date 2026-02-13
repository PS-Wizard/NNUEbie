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

    // Start position
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

    probe.set_position(&pieces);

    let iterations = 1_000_000;

    println!("\n=== Testing AccumulatorStack Performance ===\n");

    // Test 1: Make/Unmake with stack (O(1))
    println!("Test 1: Make/Unmake using AccumulatorStack (O(1))");
    let start = Instant::now();
    for i in 0..iterations {
        // Make a move: e2-e4
        probe.make_move(12, 28, Piece::WhitePawn);
        std::hint::black_box(probe.evaluate(Color::Black));

        // Unmake - O(1)!
        probe.unmake_move(12, 28, Piece::WhitePawn, None);

        if i % 100000 == 0 && i > 0 {
            print!(".");
        }
    }
    let stack_duration = start.elapsed();
    let stack_nps = iterations as f64 / stack_duration.as_secs_f64();
    println!("\nStack-based make/unmake: {:.2} ops/sec", stack_nps);
    println!(
        "Time per make/unmake cycle: {:.3} µs",
        stack_duration.as_secs_f64() * 1_000_000.0 / iterations as f64
    );

    // Test 2: Legacy update method (direct incremental)
    println!("\nTest 2: Legacy update method (direct incremental)");
    probe.set_position(&pieces);

    let start = Instant::now();
    for i in 0..iterations {
        // Apply e2-e4
        probe.update(&[(Piece::WhitePawn, 12)], &[(Piece::WhitePawn, 28)]);
        std::hint::black_box(probe.evaluate(Color::Black));

        // Apply e4-e2 (undo)
        probe.update(&[(Piece::WhitePawn, 28)], &[(Piece::WhitePawn, 12)]);

        if i % 100000 == 0 && i > 0 {
            print!(".");
        }
    }
    let legacy_duration = start.elapsed();
    let legacy_nps = iterations as f64 / legacy_duration.as_secs_f64();
    println!("\nLegacy update method: {:.2} ops/sec", legacy_nps);
    println!(
        "Time per update cycle: {:.3} µs",
        legacy_duration.as_secs_f64() * 1_000_000.0 / iterations as f64
    );

    // Test 3: Full refresh (what happens without any optimization)
    println!("\nTest 3: Full refresh on each evaluation (worst case)");
    probe.set_position(&pieces);

    let start = Instant::now();
    for i in 0..iterations {
        // Simulate: set position from scratch each time
        probe.set_position(&pieces);
        std::hint::black_box(probe.evaluate(Color::Black));

        if i % 100000 == 0 && i > 0 {
            print!(".");
        }
    }
    let refresh_duration = start.elapsed();
    let refresh_nps = iterations as f64 / refresh_duration.as_secs_f64();
    println!("\nFull refresh method: {:.2} ops/sec", refresh_nps);
    println!(
        "Time per refresh cycle: {:.3} µs",
        refresh_duration.as_secs_f64() * 1_000_000.0 / iterations as f64
    );

    // Summary
    println!("\n=== Summary ===");
    println!(
        "AccumulatorStack is {:.1}x faster than full refresh",
        refresh_nps / stack_nps
    );
    println!("\nIn a chess engine doing 5M make/unmake per second:");
    println!(
        "  - With full refresh: {:.1} seconds",
        5_000_000.0 / refresh_nps
    );
    println!(
        "  - With AccumulatorStack: {:.1} seconds",
        5_000_000.0 / stack_nps
    );
}
