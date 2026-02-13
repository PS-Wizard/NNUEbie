use crate::accumulator_stack::{AccumulatorStack, DirtyPiece};
use crate::evaluator::{
    Evaluator, NnueNetworks, BISHOP_VALUE, KNIGHT_VALUE, PAWN_VALUE, QUEEN_VALUE, ROOK_VALUE,
};
use crate::types::{Color, Piece, Square};
use std::io;
use std::sync::Arc;

pub struct NNUEProbe {
    evaluator: Evaluator,
    pieces: [Piece; 64],
    king_squares: [Square; 2],
    piece_count: usize,
    pawn_count: [i32; 2],
    non_pawn_material: [i32; 2],
    accumulator_stack: AccumulatorStack,
}

impl NNUEProbe {
    pub fn new(big_path: &str, small_path: &str) -> io::Result<Self> {
        let networks = Arc::new(NnueNetworks::new(big_path, small_path)?);
        Self::with_networks(networks)
    }

    pub fn with_networks(networks: Arc<NnueNetworks>) -> io::Result<Self> {
        let evaluator = Evaluator::new(networks);
        Ok(Self {
            evaluator,
            pieces: [Piece::None; 64],
            king_squares: [0; 2], // Default
            piece_count: 0,
            pawn_count: [0; 2],
            non_pawn_material: [0; 2],
            accumulator_stack: AccumulatorStack::new(),
        })
    }

    /// Set the root position - this does a full refresh
    pub fn set_position(&mut self, pieces: &[(Piece, Square)]) {
        // Reset state
        self.pieces = [Piece::None; 64];
        self.piece_count = 0;
        self.pawn_count = [0; 2];
        self.non_pawn_material = [0; 2];
        self.king_squares = [0; 2];
        self.accumulator_stack.reset();

        for &(piece, square) in pieces {
            self.add_piece_internal(piece, square);
        }

        self.refresh_accumulators();
    }

    fn add_piece_internal(&mut self, piece: Piece, square: Square) {
        if piece == Piece::None {
            return;
        }

        // If overwriting, remove first (though set_position clears all)
        if self.pieces[square] != Piece::None {
            // In set_position, pieces should be unique squares, but safeguard
            self.remove_piece_internal(square);
        }

        self.pieces[square] = piece;
        self.piece_count += 1;

        if let Some(color) = piece.color() {
            let c = color.index();
            if piece.piece_type() == 1 {
                // Pawn
                self.pawn_count[c] += 1;
            } else if piece.is_king() {
                self.king_squares[c] = square;
            } else {
                self.non_pawn_material[c] += self.piece_value(piece);
            }
        }
    }

    fn remove_piece_internal(&mut self, square: Square) -> Piece {
        let piece = self.pieces[square];
        if piece == Piece::None {
            return Piece::None;
        }

        self.pieces[square] = Piece::None;
        self.piece_count -= 1;

        if let Some(color) = piece.color() {
            let c = color.index();
            if piece.piece_type() == 1 {
                // Pawn
                self.pawn_count[c] -= 1;
            } else if piece.is_king() {
                // King removed
            } else {
                self.non_pawn_material[c] -= self.piece_value(piece);
            }
        }
        piece
    }

    fn piece_value(&self, piece: Piece) -> i32 {
        match piece {
            Piece::WhiteKnight | Piece::BlackKnight => KNIGHT_VALUE,
            Piece::WhiteBishop | Piece::BlackBishop => BISHOP_VALUE,
            Piece::WhiteRook | Piece::BlackRook => ROOK_VALUE,
            Piece::WhiteQueen | Piece::BlackQueen => QUEEN_VALUE,
            _ => 0,
        }
    }

    /// Make a move - pushes new state onto accumulator stack
    pub fn make_move(&mut self, from_sq: Square, to_sq: Square, piece: Piece) {
        // Build dirty piece info
        let mut dirty = DirtyPiece::new();

        let from_piece = self.pieces[from_sq];
        let to_piece = self.pieces[to_sq]; // Captured piece, if any

        // Remove piece from source
        self.remove_piece_internal(from_sq);

        // Remove captured piece from destination (if any)
        if to_piece != Piece::None {
            dirty.add_change(to_sq, to_sq, to_piece.index(), 0);
        }

        // Add piece to destination
        self.add_piece_internal(piece, to_sq);

        // Record the move
        dirty.add_change(from_sq, to_sq, from_piece.index(), piece.index());

        // Push onto stack
        self.accumulator_stack.push(&dirty);

        // Update accumulators incrementally (unless king moved)
        if piece.is_king() {
            // King moves require special handling - do full refresh
            self.refresh_accumulators();
        } else {
            // Incremental update
            self.accumulator_stack.update_incremental(
                self.king_squares,
                &self.evaluator.networks.big_net.feature_transformer,
                &self.evaluator.networks.small_net.feature_transformer,
            );
        }
    }

    /// Unmake a move - pops state from accumulator stack (O(1)!)
    pub fn unmake_move(
        &mut self,
        from_sq: Square,
        to_sq: Square,
        from_piece: Piece,
        captured_piece: Option<Piece>,
    ) {
        // Restore the piece state (inverse of make_move)
        self.remove_piece_internal(to_sq);

        if let Some(captured) = captured_piece {
            self.add_piece_internal(captured, to_sq);
        }

        self.add_piece_internal(from_piece, from_sq);

        // Pop from stack - O(1)!
        self.accumulator_stack.pop();
    }

    /// Legacy update method - applies changes directly to current accumulators
    /// Does NOT use the stack - for one-off evaluations only
    pub fn update(&mut self, removed: &[(Piece, Square)], added: &[(Piece, Square)]) {
        let mut removed_mapped: Vec<(usize, usize)> = Vec::with_capacity(removed.len());
        let mut added_mapped: Vec<(usize, usize)> = Vec::with_capacity(added.len());

        // Track if king moved
        let mut king_moved = false;

        // Apply removals
        for &(piece, square) in removed {
            self.remove_piece_internal(square);
            removed_mapped.push((square, piece.index()));
            if piece.is_king() {
                king_moved = true;
            }
        }

        // Apply additions
        for &(piece, square) in added {
            self.add_piece_internal(piece, square);
            added_mapped.push((square, piece.index()));
            if piece.is_king() {
                king_moved = true;
            }
        }

        if king_moved {
            // Full refresh required
            self.refresh_accumulators();
        } else {
            // Direct incremental update on current stack position
            let state = self.accumulator_stack.mut_latest();

            state.acc_big.update_with_ksq(
                &added_mapped,
                &removed_mapped,
                self.king_squares,
                &self.evaluator.networks.big_net.feature_transformer,
            );
            state.acc_small.update_with_ksq(
                &added_mapped,
                &removed_mapped,
                self.king_squares,
                &self.evaluator.networks.small_net.feature_transformer,
            );
        }
    }

    fn refresh_accumulators(&mut self) {
        // Collect all pieces
        let mut pieces_idx = Vec::with_capacity(self.piece_count);
        for sq in 0..64 {
            let p = self.pieces[sq];
            if p != Piece::None {
                pieces_idx.push((sq, p.index()));
            }
        }

        self.accumulator_stack.refresh(
            &pieces_idx,
            self.king_squares,
            &self.evaluator.networks.big_net.feature_transformer,
            &self.evaluator.networks.small_net.feature_transformer,
        );
    }

    pub fn evaluate(&mut self, side_to_move: Color) -> i32 {
        let stm = side_to_move.index();
        let simple_eval = PAWN_VALUE * (self.pawn_count[stm] - self.pawn_count[1 - stm])
            + (self.non_pawn_material[stm] - self.non_pawn_material[1 - stm]);

        let use_small = simple_eval.abs() > 962;

        let bucket = if self.piece_count > 0 {
            (self.piece_count - 1) / 4
        } else {
            0
        };
        let bucket = bucket.min(7);

        let mut nnue_val;
        let mut psqt_val;
        let mut positional_val;

        // Get latest accumulator state from stack
        let latest_state = self.accumulator_stack.latest();

        if use_small {
            let (psqt, pos) = self.evaluator.networks.small_net.evaluate(
                &latest_state.acc_small,
                bucket,
                stm,
                &mut self.evaluator.scratch_small,
            );
            nnue_val = (125 * psqt + 131 * pos) / 128;
            psqt_val = psqt;
            positional_val = pos;

            if nnue_val.abs() < 236 {
                // Use big network
                let (psqt_b, pos_b) = self.evaluator.networks.big_net.evaluate(
                    &latest_state.acc_big,
                    bucket,
                    stm,
                    &mut self.evaluator.scratch_big,
                );
                nnue_val = (125 * psqt_b + 131 * pos_b) / 128;
                psqt_val = psqt_b;
                positional_val = pos_b;
            }
        } else {
            // Use big network
            let (psqt, pos) = self.evaluator.networks.big_net.evaluate(
                &latest_state.acc_big,
                bucket,
                stm,
                &mut self.evaluator.scratch_big,
            );
            nnue_val = (125 * psqt + 131 * pos) / 128;
            psqt_val = psqt;
            positional_val = pos;
        }

        let nnue_complexity = (psqt_val - positional_val).abs();
        nnue_val -= nnue_val * nnue_complexity / 18000;

        let material = 535 * (self.pawn_count[0] + self.pawn_count[1])
            + (self.non_pawn_material[0] + self.non_pawn_material[1]);

        let optimism = 0;
        let v = (nnue_val * (77777 + material) + optimism * (7777 + material)) / 77777;

        // Clamp to avoid tablebase range overlaps
        v.clamp(-31753, 31753)
    }
}
