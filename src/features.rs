use crate::types::{Color, Piece};

pub const SQUARE_NB: usize = 64;
pub const PIECE_NB: usize = 16; // Stockfish definition including None
pub const COLOR_NB: usize = 2;

// Re-export for convenience
pub use crate::types::{Color as ColorType, Piece as PieceType, Square as SquareIndex};

// Color and Piece constants for indexing (convert Color/Piece enums to usize)
pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

// Piece type indices (1-6)
pub const PAWN: usize = 1;
pub const KNIGHT: usize = 2;
pub const BISHOP: usize = 3;
pub const ROOK: usize = 4;
pub const QUEEN: usize = 5;
pub const KING: usize = 6;

// Helper to get white/black index
pub const fn white() -> usize {
    0
}
pub const fn black() -> usize {
    1
}

// Helper to get piece type index (1-6)
pub const fn piece_type_idx(pt: Piece) -> usize {
    match pt {
        Piece::WhitePawn | Piece::BlackPawn => 1,
        Piece::WhiteKnight | Piece::BlackKnight => 2,
        Piece::WhiteBishop | Piece::BlackBishop => 3,
        Piece::WhiteRook | Piece::BlackRook => 4,
        Piece::WhiteQueen | Piece::BlackQueen => 5,
        Piece::WhiteKing | Piece::BlackKing => 6,
        Piece::None => 0,
    }
}

// Convert Color enum to index
pub const fn color_idx(c: Color) -> usize {
    match c {
        Color::White => 0,
        Color::Black => 1,
    }
}

// Convert Piece to nnue piece index (0-15)
pub const fn piece_index(p: Piece) -> usize {
    p as usize
}

// Piece to NNUE index mapping
// 11 types * 64 squares
const PS_W_PAWN: usize = 0;
const PS_B_PAWN: usize = SQUARE_NB;
const PS_W_KNIGHT: usize = 2 * SQUARE_NB;
const PS_B_KNIGHT: usize = 3 * SQUARE_NB;
const PS_W_BISHOP: usize = 4 * SQUARE_NB;
const PS_B_BISHOP: usize = 5 * SQUARE_NB;
const PS_W_ROOK: usize = 6 * SQUARE_NB;
const PS_B_ROOK: usize = 7 * SQUARE_NB;
const PS_W_QUEEN: usize = 8 * SQUARE_NB;
const PS_B_QUEEN: usize = 9 * SQUARE_NB;
const PS_KING: usize = 10 * SQUARE_NB;
const PS_NONE: usize = 0;

pub const PIECE_SQUARE_INDEX: [[usize; 16]; 2] = [
    // White Perspective
    [
        PS_NONE,
        PS_W_PAWN,
        PS_W_KNIGHT,
        PS_W_BISHOP,
        PS_W_ROOK,
        PS_W_QUEEN,
        PS_KING,
        PS_NONE,
        PS_NONE,
        PS_B_PAWN,
        PS_B_KNIGHT,
        PS_B_BISHOP,
        PS_B_ROOK,
        PS_B_QUEEN,
        PS_KING,
        PS_NONE,
    ],
    // Black Perspective
    [
        PS_NONE,
        PS_B_PAWN,
        PS_B_KNIGHT,
        PS_B_BISHOP,
        PS_B_ROOK,
        PS_B_QUEEN,
        PS_KING,
        PS_NONE,
        PS_NONE,
        PS_W_PAWN,
        PS_W_KNIGHT,
        PS_W_BISHOP,
        PS_W_ROOK,
        PS_W_QUEEN,
        PS_KING,
        PS_NONE,
    ],
];

// Encoded KingBuckets from C++ code
// B(v) = v * 11 * 64 (PS_NB)
const PS_NB: usize = 11 * 64;
const fn b(v: usize) -> usize {
    v * PS_NB
}

pub const KING_BUCKETS: [[usize; 64]; 2] = [
    // White
    [
        b(28),
        b(29),
        b(30),
        b(31),
        b(31),
        b(30),
        b(29),
        b(28),
        b(24),
        b(25),
        b(26),
        b(27),
        b(27),
        b(26),
        b(25),
        b(24),
        b(20),
        b(21),
        b(22),
        b(23),
        b(23),
        b(22),
        b(21),
        b(20),
        b(16),
        b(17),
        b(18),
        b(19),
        b(19),
        b(18),
        b(17),
        b(16),
        b(12),
        b(13),
        b(14),
        b(15),
        b(15),
        b(14),
        b(13),
        b(12),
        b(8),
        b(9),
        b(10),
        b(11),
        b(11),
        b(10),
        b(9),
        b(8),
        b(4),
        b(5),
        b(6),
        b(7),
        b(7),
        b(6),
        b(5),
        b(4),
        b(0),
        b(1),
        b(2),
        b(3),
        b(3),
        b(2),
        b(1),
        b(0),
    ],
    // Black
    [
        b(0),
        b(1),
        b(2),
        b(3),
        b(3),
        b(2),
        b(1),
        b(0),
        b(4),
        b(5),
        b(6),
        b(7),
        b(7),
        b(6),
        b(5),
        b(4),
        b(8),
        b(9),
        b(10),
        b(11),
        b(11),
        b(10),
        b(9),
        b(8),
        b(12),
        b(13),
        b(14),
        b(15),
        b(15),
        b(14),
        b(13),
        b(12),
        b(16),
        b(17),
        b(18),
        b(19),
        b(19),
        b(18),
        b(17),
        b(16),
        b(20),
        b(21),
        b(22),
        b(23),
        b(23),
        b(22),
        b(21),
        b(20),
        b(24),
        b(25),
        b(26),
        b(27),
        b(27),
        b(26),
        b(25),
        b(24),
        b(28),
        b(29),
        b(30),
        b(31),
        b(31),
        b(30),
        b(29),
        b(28),
    ],
];

const SQ_H1: usize = 7;
const SQ_A1: usize = 0;
const SQ_H8: usize = 63;
const SQ_A8: usize = 56;

pub const ORIENT_TBL: [[usize; 64]; 2] = [
    // White
    [
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1,
        SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1,
        SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1,
        SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    ],
    // Black
    [
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8,
        SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8,
        SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8,
        SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8,
        SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
    ],
];

pub fn make_index(perspective: usize, s: usize, pc: usize, ksq: usize) -> usize {
    let orient = ORIENT_TBL[perspective][ksq];
    let piece_offset = PIECE_SQUARE_INDEX[perspective][pc];
    let bucket_offset = KING_BUCKETS[perspective][ksq];

    (s ^ orient) + piece_offset + bucket_offset
}
