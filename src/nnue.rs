use crate::evaluator::{
    Evaluator, NnueNetworks, BISHOP_VALUE, KNIGHT_VALUE, PAWN_VALUE, QUEEN_VALUE, ROOK_VALUE,
};
use crate::network::ScratchBuffer;
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
    scratch_big: Option<ScratchBuffer>,
    scratch_small: Option<ScratchBuffer>,
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
            scratch_big: None,
            scratch_small: None,
        })
    }

    pub fn set_position(&mut self, pieces: &[(Piece, Square)]) {
        // Reset state
        self.pieces = [Piece::None; 64];
        self.piece_count = 0;
        self.pawn_count = [0; 2];
        self.non_pawn_material = [0; 2];
        self.king_squares = [0; 2];

        for &(piece, square) in pieces {
            self.add_piece_internal(piece, square);
        }

        self.refresh();
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

    pub fn update(&mut self, removed: &[(Piece, Square)], added: &[(Piece, Square)]) {
        let mut king_moved = false;

        // Apply removals
        for &(piece, square) in removed {
            let _removed_piece = self.remove_piece_internal(square);
            // Debug assertion could go here
            if piece.is_king() {
                king_moved = true;
            }
        }

        // Apply additions
        for &(piece, square) in added {
            self.add_piece_internal(piece, square);
            if piece.is_king() {
                king_moved = true;
            }
        }

        if king_moved {
            self.refresh();
        } else {
            // Incremental update
            // Map pieces to (Square, PieceIndex)
            // Allocating Vecs for now - Phase 3 will optimize
            let removed_mapped: Vec<(usize, usize)> =
                removed.iter().map(|&(p, s)| (s, p.index())).collect();
            let added_mapped: Vec<(usize, usize)> =
                added.iter().map(|&(p, s)| (s, p.index())).collect();

            let ft_big = &self.evaluator.networks.big_net.feature_transformer;
            self.evaluator.acc_big.update_with_ksq(
                &added_mapped,
                &removed_mapped,
                self.king_squares,
                ft_big,
            );

            let ft_small = &self.evaluator.networks.small_net.feature_transformer;
            self.evaluator.acc_small.update_with_ksq(
                &added_mapped,
                &removed_mapped,
                self.king_squares,
                ft_small,
            );
        }
    }

    pub fn refresh(&mut self) {
        // Collect all pieces
        let mut pieces_idx = Vec::with_capacity(self.piece_count);
        for sq in 0..64 {
            let p = self.pieces[sq];
            if p != Piece::None {
                pieces_idx.push((sq, p.index()));
            }
        }

        let ft_big = &self.evaluator.networks.big_net.feature_transformer;
        self.evaluator
            .acc_big
            .refresh(&pieces_idx, self.king_squares, ft_big);

        let ft_small = &self.evaluator.networks.small_net.feature_transformer;
        self.evaluator
            .acc_small
            .refresh(&pieces_idx, self.king_squares, ft_small);
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

        if use_small {
            if self.scratch_small.is_none() {
                let half_dims = self
                    .evaluator
                    .networks
                    .small_net
                    .feature_transformer
                    .half_dims;
                self.scratch_small = Some(ScratchBuffer::new(half_dims));
            }
            let scratch = self.scratch_small.as_mut().unwrap();
            let (psqt, pos) = self.evaluator.networks.small_net.evaluate(
                &self.evaluator.acc_small,
                bucket,
                stm,
                scratch,
            );
            nnue_val = (125 * psqt + 131 * pos) / 128;
            psqt_val = psqt;
            positional_val = pos;

            if nnue_val.abs() < 236 {
                // Use big
                if self.scratch_big.is_none() {
                    let half_dims = self
                        .evaluator
                        .networks
                        .big_net
                        .feature_transformer
                        .half_dims;
                    self.scratch_big = Some(ScratchBuffer::new(half_dims));
                }
                let scratch_big = self.scratch_big.as_mut().unwrap();
                let (psqt_b, pos_b) = self.evaluator.networks.big_net.evaluate(
                    &self.evaluator.acc_big,
                    bucket,
                    stm,
                    scratch_big,
                );
                nnue_val = (125 * psqt_b + 131 * pos_b) / 128;
                psqt_val = psqt_b;
                positional_val = pos_b;
            }
        } else {
            if self.scratch_big.is_none() {
                let half_dims = self
                    .evaluator
                    .networks
                    .big_net
                    .feature_transformer
                    .half_dims;
                self.scratch_big = Some(ScratchBuffer::new(half_dims));
            }
            let scratch = self.scratch_big.as_mut().unwrap();
            let (psqt, pos) = self.evaluator.networks.big_net.evaluate(
                &self.evaluator.acc_big,
                bucket,
                stm,
                scratch,
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
