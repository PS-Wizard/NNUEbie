use nnuebie::uci::{calculate_material, to_centipawns};
use nnuebie::{Evaluator, NnueNetworks, BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use std::sync::Arc;

fn parse_fen(fen: &str) -> (Vec<(usize, usize, usize)>, usize, i32) {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let board_str = parts[0];
    let side_str = parts[1];
    let rule50_str = if parts.len() > 4 { parts[4] } else { "0" };
    let rule50: i32 = rule50_str.parse().unwrap_or(0);

    let mut pieces = Vec::new();
    let mut rank = 7;
    let mut file = 0;

    for c in board_str.chars() {
        if c == '/' {
            rank -= 1;
            file = 0;
        } else if c.is_digit(10) {
            file += c.to_digit(10).unwrap() as usize;
        } else {
            let color = if c.is_uppercase() { WHITE } else { BLACK };
            let pt = match c.to_ascii_lowercase() {
                'p' => PAWN,
                'n' => KNIGHT,
                'b' => BISHOP,
                'r' => ROOK,
                'q' => QUEEN,
                'k' => KING,
                _ => panic!("Invalid piece char: {}", c),
            };
            let sq = rank * 8 + file;
            pieces.push((sq, pt, color));
            file += 1;
        }
    }

    let side = if side_str == "w" { WHITE } else { BLACK };
    (pieces, side, rule50)
}

fn main() {
    let big_path = "archive/nnue/networks/nn-1c0000000000.nnue";
    let small_path = "archive/nnue/networks/nn-37f18f62d772.nnue";

    println!("Loading networks...");
    let networks =
        Arc::new(NnueNetworks::new(big_path, small_path).expect("Failed to load networks"));
    let mut eval = Evaluator::new(networks);

    // Expected values from Stockfish "eval" command output (Final evaluation, White side)
    // Tolerance is 0
    let test_cases = vec![
        (
            "Startpos",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            7,
        ),
        (
            "King Triggers Refresh",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            -20,
        ),
        (
            "e4",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            37,
        ),
        (
            "No Queen",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
            -522,
        ), // -5.22
        (
            "Opening",
            "r1bqkb1r/pppp1ppp/2n2n2/3Pp3/4P3/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 0 1",
            113,
        ), // +1.13
        (
            "Middlegame",
            "r1bq1rk1/ppp1npbp/2np2p1/4p3/2P4N/2NP2P1/PP2PPBP/R1BQ1RK1 w - - 0 1",
            4,
        ), // +0.04
        (
            "Middlegame",
            "r1bq1rk1/1pp2pbN/2np4/4p3/7N/3P2P1/1P2PPBP/R1BQ1RK1 w - - 0 1",
            389,
        ), // +0.04
    ];

    for (name, fen, expected_cp) in test_cases {
        let (pieces, side, rule50) = parse_fen(fen);
        let score_internal = eval.evaluate(&pieces, side, rule50);
        let material = calculate_material(&pieces);
        let score_cp = to_centipawns(score_internal, material);

        // Convert to White perspective for comparison with Stockfish "white side" output
        let score_cp_white = if side == BLACK { -score_cp } else { score_cp };

        println!("Position: {}", name);
        println!("FEN: {}", fen);
        println!("Material Factor (Count): {}", material);
        println!("Rule50: {}", rule50);
        println!("Internal Score: {} (Side to move)", score_internal);
        println!("Centipawn Score: {} (Side to move)", score_cp);
        println!("Centipawn Score: {} (White side)", score_cp_white);
        println!("Expected CP: {} (White side)", expected_cp);

        let diff = (score_cp_white - expected_cp).abs();
        if diff <= 0 {
            println!("Result: PASS (Diff {})", diff);
        } else {
            println!("Result: FAIL (Diff {})", diff);
        }
        println!("--------------------------------------------------");
    }
}
