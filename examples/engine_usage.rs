use nnuebie::{Color, NNUEProbe, Piece, Square};
use std::io;

/// Helper function to parse a FEN string into a list of pieces and the side to move.
/// This adapts FEN characters to the library's internal `Piece` and `Color` types.
fn parse_fen(fen: &str) -> (Vec<(Piece, Square)>, Color) {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let board_str = parts[0];
    let side_str = parts[1];

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
            let piece = match c {
                'P' => Piece::WhitePawn,
                'N' => Piece::WhiteKnight,
                'B' => Piece::WhiteBishop,
                'R' => Piece::WhiteRook,
                'Q' => Piece::WhiteQueen,
                'K' => Piece::WhiteKing,
                'p' => Piece::BlackPawn,
                'n' => Piece::BlackKnight,
                'b' => Piece::BlackBishop,
                'r' => Piece::BlackRook,
                'q' => Piece::BlackQueen,
                'k' => Piece::BlackKing,
                _ => panic!("Invalid piece char: {}", c),
            };
            let sq = rank * 8 + file;
            pieces.push((piece, sq));
            file += 1;
        }
    }

    let side = if side_str == "w" {
        Color::White
    } else {
        Color::Black
    };
    (pieces, side)
}

/// Calculate simple material count for UCI conversion (1, 3, 3, 5, 9).
/// This is required if you want to convert the internal NNUE score to standard centipawns.
fn calculate_material(pieces: &[(Piece, Square)]) -> i32 {
    let mut material = 0;
    for (piece, _) in pieces {
        let val = match piece {
            Piece::WhitePawn | Piece::BlackPawn => 1,
            Piece::WhiteKnight | Piece::BlackKnight => 3,
            Piece::WhiteBishop | Piece::BlackBishop => 3,
            Piece::WhiteRook | Piece::BlackRook => 5,
            Piece::WhiteQueen | Piece::BlackQueen => 9,
            _ => 0,
        };
        material += val;
    }
    material
}

fn main() -> io::Result<()> {
    // 1. Initialize the NNUE Probe with paths to network files
    // You must provide the paths to the Big and Small networks.
    let big_path = "archive/nnue/networks/nn-1c0000000000.nnue";
    let small_path = "archive/nnue/networks/nn-37f18f62d772.nnue";

    println!("Loading NNUE networks...");
    // The Probe wrapper handles state, accumulators, and loading.
    let mut probe = NNUEProbe::new(big_path, small_path)?;

    // 2. Set position from FEN
    // Example: Start Position
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    println!("Setting position: {}", fen);

    let (pieces, side) = parse_fen(fen);

    // `set_position` automatically handles a full refresh of the NNUE accumulators.
    // This is required when setting up a new board state from scratch.
    probe.set_position(&pieces, 0);

    // 3. Get Evaluation
    // `evaluate` returns the internal raw score from the perspective of `side`.
    let score_internal = probe.evaluate(side);

    // Convert to Centipawns (optional, usually preferred for UCI engines).
    // The conversion depends on total non-pawn material.
    let material_count = calculate_material(&pieces);
    let score_cp = nnuebie::uci::to_centipawns(score_internal, material_count);

    println!("Internal Score: {}", score_internal);
    println!("Evaluation (centipawns): {}", score_cp);

    // 4. Incremental Update Example (e2 -> e4)
    println!("\nMaking move: e2e4");

    // Define the move (White Pawn from e2 to e4)
    let from_sq = 12; // e2
    let to_sq = 28; // e4
    let piece = Piece::WhitePawn;

    // Prepare lists of pieces removed and added for the update
    let removed = vec![(piece, from_sq)];
    let added = vec![(piece, to_sq)];

    // Apply the update
    // The library's `update` method is smart:
    // - It will try to perform a fast incremental update of the accumulators.
    // - If a King moved (which invalidates accumulators relative to that King),
    //   it automatically falls back to a full refresh.
    // - You do not need to manually check for refresh conditions.
    probe.update(&removed, &added);

    // Evaluate new position (now Black to move)
    let side_to_move = Color::Black;
    let score_after_internal = probe.evaluate(side_to_move);

    // Note: Incremental updates to material count would also be handled by an engine here.
    // Since e2e4 captures nothing, material count is unchanged.
    let score_after_cp = nnuebie::uci::to_centipawns(score_after_internal, material_count);

    println!(
        "Evaluation after e2e4 (Black to move): {} cp",
        score_after_cp
    );

    Ok(())
}
