pub mod accumulator;
pub mod accumulator_stack;
pub mod aligned;
pub mod evaluator;
pub mod feature_transformer;
pub mod features;
pub mod finny_tables;
pub mod layers;
pub mod loader;
pub mod network;
pub mod nnue;
pub mod types;
pub mod uci;

#[cfg(test)]
mod tests;

pub use accumulator::Accumulator;
pub use accumulator_stack::{AccumulatorStack, AccumulatorState, DirtyPiece};
pub use evaluator::{Evaluator, NnueNetworks};
pub use feature_transformer::FeatureTransformer;
pub use finny_tables::FinnyTables;
pub use layers::Layer;
pub use network::Network;
pub use nnue::NNUEProbe;
pub use types::{Color, Piece, Square};

pub use features::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};

pub const VERSION: u32 = 0x7AF32F20;

// From nnue_common.h
pub const OUTPUT_SCALE: i32 = 16;
pub const WEIGHT_SCALE_BITS: i32 = 6;
