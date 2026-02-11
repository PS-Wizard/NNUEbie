use nnue_rs::{Evaluator, BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use std::time::Instant;

fn main() {
    let big_path = "archive/nnue/networks/nn-1c0000000000.nnue";
    let small_path = "archive/nnue/networks/nn-37f18f62d772.nnue";

    println!("Loading networks...");
    // Check if files exist relative to current dir, otherwise try relative to crate root if running via cargo run
    // Assuming running from crate root
    let mut eval = match Evaluator::new(big_path, small_path) {
        Ok(e) => e,
        Err(_) => {
            // Try ../../ for when running from bin? No, cargo run runs from workspace root.
            Evaluator::new(
                "../../archive/nnue/networks/nn-1c0000000000.nnue",
                "../../archive/nnue/networks/nn-37f18f62d772.nnue",
            )
            .expect("Failed to load networks")
        }
    };

    // Position: Startpos
    // rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    let pieces = vec![
        (0, ROOK, WHITE),
        (1, KNIGHT, WHITE),
        (2, BISHOP, WHITE),
        (3, QUEEN, WHITE),
        (4, KING, WHITE),
        (5, BISHOP, WHITE),
        (6, KNIGHT, WHITE),
        (7, ROOK, WHITE),
        (8, PAWN, WHITE),
        (9, PAWN, WHITE),
        (10, PAWN, WHITE),
        (11, PAWN, WHITE),
        (12, PAWN, WHITE),
        (13, PAWN, WHITE),
        (14, PAWN, WHITE),
        (15, PAWN, WHITE),
        (48, PAWN, BLACK),
        (49, PAWN, BLACK),
        (50, PAWN, BLACK),
        (51, PAWN, BLACK),
        (52, PAWN, BLACK),
        (53, PAWN, BLACK),
        (54, PAWN, BLACK),
        (55, PAWN, BLACK),
        (56, ROOK, BLACK),
        (57, KNIGHT, BLACK),
        (58, BISHOP, BLACK),
        (59, QUEEN, BLACK),
        (60, KING, BLACK),
        (61, BISHOP, BLACK),
        (62, KNIGHT, BLACK),
        (63, ROOK, BLACK),
    ];
    let side = WHITE;

    // Warmup
    for _ in 0..100 {
        eval.evaluate(&pieces, side);
    }

    let iterations = 1_000_000;
    println!("Benchmarking {} evaluations...", iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        // We use black_box to prevent optimization if possible
        std::hint::black_box(eval.evaluate(&pieces, side));
    }
    let duration = start.elapsed();

    let nps = iterations as f64 / duration.as_secs_f64();
    println!("Full Refresh Throughput: {:.2} evaluations/sec", nps);

    // Benchmark Incremental Update
    println!("\nBenchmarking Incremental Update...");
    let ft = &eval.big_net.feature_transformer;
    let ksq = [4, 60];
    let mut acc = nnue_rs::accumulator::Accumulator::new(ft.half_dims);

    // Convert pieces to (Square, PieceWithColor) format expected by Accumulator
    let simple_pieces: Vec<(usize, usize)> =
        pieces.iter().map(|&(sq, pc, c)| (sq, pc + 8 * c)).collect();

    acc.refresh(&simple_pieces, ksq, ft);

    // Move Pawn E2 -> E4
    // Remove (12, PAWN) (White Pawn = PAWN + 0 = 1)
    // Add (28, PAWN)
    // Wait, PAWN constant is 1.
    // White Pawn is 1.
    // Black Pawn is 9 (1 + 8).

    // In previous code I used (12, PAWN) assuming PAWN was the correct value passed to refresh.
    // But Evaluator passes (color * 8) + pc.
    // PAWN=1.
    // So White Pawn = 1.
    // Black Pawn = 9.

    let removed = vec![(12, PAWN)]; // White Pawn
    let added = vec![(28, PAWN)]; // White Pawn

    let iterations_inc = 1_000_000;

    let start = Instant::now();
    for i in 0..iterations_inc {
        // Toggle move to keep state valid-ish or just repeat same update?
        // Repeating same update is fine for throughput measurement of the function itself.
        // But accumulation values will explode/drift.
        // For performance check, it doesn't matter if values are correct chess-wise, just the computation cost.
        // But to be cleaner, let's swap added/removed every iteration.

        if i % 2 == 0 {
            acc.update_with_ksq(&added, &removed, ksq, ft);
        } else {
            acc.update_with_ksq(&removed, &added, ksq, ft);
        }

        // Also run network evaluation (forward pass) because "evaluate" includes it
        // Evaluator::evaluate does: refresh + network.evaluate
        // So we should measure: update + network.evaluate
        let bucket = (pieces.len() - 1) / 4;
        std::hint::black_box(eval.big_net.evaluate(&acc, bucket, side));
    }
    let duration = start.elapsed();
    let nps_inc = iterations_inc as f64 / duration.as_secs_f64();
    println!(
        "Incremental Update Throughput: {:.2} evaluations/sec",
        nps_inc
    );
    println!("Speedup: {:.2}x", nps_inc / nps);
}
