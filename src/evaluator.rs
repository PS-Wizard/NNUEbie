use crate::accumulator::Accumulator;
use crate::features::{BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK};
use crate::network::{Network, ScratchBuffer};
use std::io;
use std::sync::Arc;

// Piece Values
pub const PAWN_VALUE: i32 = 208;
pub const KNIGHT_VALUE: i32 = 781;
pub const BISHOP_VALUE: i32 = 825;
pub const ROOK_VALUE: i32 = 1276;
pub const QUEEN_VALUE: i32 = 2538;

pub struct NnueNetworks {
    pub big_net: Network,
    pub small_net: Network,
}

impl NnueNetworks {
    pub fn new(big_path: &str, small_path: &str) -> io::Result<Self> {
        let big_net = Network::load(big_path, true)?;
        let small_net = Network::load(small_path, false)?;
        Ok(Self { big_net, small_net })
    }
}

pub struct Evaluator {
    pub networks: Arc<NnueNetworks>,
    pub acc_big: Accumulator,
    pub acc_small: Accumulator,
    scratch_big: Option<ScratchBuffer>,
    scratch_small: Option<ScratchBuffer>,
}

impl Evaluator {
    pub fn new(networks: Arc<NnueNetworks>) -> Self {
        let acc_big = Accumulator::new(networks.big_net.feature_transformer.half_dims);
        let acc_small = Accumulator::new(networks.small_net.feature_transformer.half_dims);

        Self {
            networks,
            acc_big,
            acc_small,
            scratch_big: None,
            scratch_small: None,
        }
    }

    pub fn evaluate(&mut self, pieces: &[(usize, usize, usize)], side_to_move: usize) -> i32 {
        // Calculate auxiliary info
        let mut pawn_count = [0; 2];
        let mut non_pawn_material = [0; 2];
        let mut piece_count = 0;
        let mut ksq = [0; 2];

        // Simplified pieces list for refresh
        let mut simple_pieces = Vec::with_capacity(pieces.len());

        for &(sq, pc, color) in pieces {
            simple_pieces.push((sq, (color * 8) + pc)); // 0..15 format
            piece_count += 1;

            if pc == PAWN {
                pawn_count[color] += 1;
            } else if pc == KING {
                ksq[color] = sq;
            } else {
                let val = match pc {
                    KNIGHT => KNIGHT_VALUE,
                    BISHOP => BISHOP_VALUE,
                    ROOK => ROOK_VALUE,
                    QUEEN => QUEEN_VALUE,
                    _ => 0,
                };
                non_pawn_material[color] += val;
            }
        }

        // Simple Eval
        let simple_eval = PAWN_VALUE
            * (pawn_count[side_to_move] as i32 - pawn_count[1 - side_to_move] as i32)
            + (non_pawn_material[side_to_move] - non_pawn_material[1 - side_to_move]);

        let use_small = simple_eval.abs() > 962;

        // Refresh accumulators
        // Stockfish refreshes both if needed or uses lazy logic.

        let mut nnue_val;
        let psqt_val;
        let positional_val;

        // Compute Bucket
        // bucket = (all_pieces - 1) / 4
        let bucket = if piece_count > 0 {
            (piece_count - 1) / 4
        } else {
            0
        };
        // Clamp bucket to 0..7
        let bucket = bucket.min(7);

        if use_small {
            self.acc_small.refresh(
                &simple_pieces,
                ksq,
                &self.networks.small_net.feature_transformer,
            );

            if self.scratch_small.is_none() {
                let half_dims = self.networks.small_net.feature_transformer.half_dims;
                self.scratch_small = Some(ScratchBuffer::new(half_dims));
            }
            let scratch = self.scratch_small.as_mut().unwrap();

            let (psqt, pos) =
                self.networks
                    .small_net
                    .evaluate(&self.acc_small, bucket, side_to_move, scratch);
            nnue_val = (125 * psqt + 131 * pos) / 128;
            psqt_val = psqt;
            positional_val = pos;

            // Correction check
            if nnue_val.abs() < 236 {
                // Re-evaluate with Big
                self.acc_big.refresh(
                    &simple_pieces,
                    ksq,
                    &self.networks.big_net.feature_transformer,
                );

                if self.scratch_big.is_none() {
                    let half_dims = self.networks.big_net.feature_transformer.half_dims;
                    self.scratch_big = Some(ScratchBuffer::new(half_dims));
                }
                let scratch_big = self.scratch_big.as_mut().unwrap();

                let (psqt_b, pos_b) = self.networks.big_net.evaluate(
                    &self.acc_big,
                    bucket,
                    side_to_move,
                    scratch_big,
                );
                nnue_val = (125 * psqt_b + 131 * pos_b) / 128;
            } else {
                // Keep Small Net values
            }
        } else {
            self.acc_big.refresh(
                &simple_pieces,
                ksq,
                &self.networks.big_net.feature_transformer,
            );

            if self.scratch_big.is_none() {
                let half_dims = self.networks.big_net.feature_transformer.half_dims;
                self.scratch_big = Some(ScratchBuffer::new(half_dims));
            }
            let scratch = self.scratch_big.as_mut().unwrap();

            let (psqt, pos) =
                self.networks
                    .big_net
                    .evaluate(&self.acc_big, bucket, side_to_move, scratch);
            nnue_val = (125 * psqt + 131 * pos) / 128;
            psqt_val = psqt;
            positional_val = pos;
        }

        // Final Scaling
        let nnue_complexity = (psqt_val - positional_val).abs();
        // Optimism (0 passed in evaluate, but logic: optimism += optimism * complexity / 468) -> 0.
        // nnue -= nnue * nnueComplexity / 18000
        nnue_val -= nnue_val * nnue_complexity / 18000;

        let material = 535 * (pawn_count[0] + pawn_count[1]) as i32
            + (non_pawn_material[0] + non_pawn_material[1]);

        let optimism = 0; // Default

        let v = (nnue_val * (77777 + material) + optimism * (7777 + material)) / 77777;

        // Rule50 dampening omitted (assume rule50 = 0)
        // v -= v * rule50 / 212;

        // Clamp to TB win/loss range
        v.clamp(-31753, 31753)
    }
}
