pub mod accumulator;
pub mod evaluator;
pub mod feature_transformer;
pub mod features;
pub mod layers;
pub mod loader;
pub mod network;
pub mod uci;

pub use evaluator::Evaluator;

pub use features::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};

pub const VERSION: u32 = 0x7AF32F20;

// From nnue_common.h
pub const OUTPUT_SCALE: i32 = 16;
pub const WEIGHT_SCALE_BITS: i32 = 6;
