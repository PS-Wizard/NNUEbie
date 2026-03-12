use crate::accumulator_stack::{AccumulatorStack, DirtyPiece};
use crate::finny_tables::FinnyTables;
use crate::network::{
    NnueNetworks, ScratchBuffer, BISHOP_VALUE, KNIGHT_VALUE, PAWN_VALUE, QUEEN_VALUE, ROOK_VALUE,
};
use crate::piece_list::{collect_pieces_from, PieceList, PIECE_LIST_CAPACITY};
use crate::types::{Color, Piece, Square};
use std::error::Error;
use std::fmt;
use std::io;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaChange {
    pub from: Square,
    pub to: Square,
    pub piece_from: Piece,
    pub piece_to: Piece,
}

impl DeltaChange {
    pub const fn new(from: Square, to: Square, piece_from: Piece, piece_to: Piece) -> Self {
        Self {
            from,
            to,
            piece_from,
            piece_to,
        }
    }

    pub const fn move_piece(from: Square, to: Square, piece_from: Piece, piece_to: Piece) -> Self {
        Self::new(from, to, piece_from, piece_to)
    }

    pub const fn removal(square: Square, piece: Piece) -> Self {
        Self::new(square, square, piece, Piece::None)
    }

    pub const fn addition(square: Square, piece: Piece) -> Self {
        Self::new(square, square, Piece::None, piece)
    }

    const fn is_empty(self) -> bool {
        matches!(self.piece_from, Piece::None) && matches!(self.piece_to, Piece::None)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaError {
    EmptyChange,
    TooManyChanges,
}

impl fmt::Display for DeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeltaError::EmptyChange => f.write_str("delta change must modify at least one piece"),
            DeltaError::TooManyChanges => f.write_str("move delta exceeds the 3-change limit"),
        }
    }
}

impl Error for DeltaError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoveDelta {
    changes: [DeltaChange; 3],
    len: usize,
    next_rule50: i32,
}

impl Default for MoveDelta {
    fn default() -> Self {
        Self::new(0)
    }
}

impl MoveDelta {
    pub const MAX_CHANGES: usize = 3;

    pub const fn new(next_rule50: i32) -> Self {
        Self {
            changes: [DeltaChange::new(0, 0, Piece::None, Piece::None); 3],
            len: 0,
            next_rule50,
        }
    }

    pub const fn null(next_rule50: i32) -> Self {
        Self::new(next_rule50)
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn next_rule50(&self) -> i32 {
        self.next_rule50
    }

    pub fn set_next_rule50(&mut self, next_rule50: i32) {
        self.next_rule50 = next_rule50;
    }

    pub fn push(&mut self, change: DeltaChange) -> Result<(), DeltaError> {
        if change.is_empty() {
            return Err(DeltaError::EmptyChange);
        }
        if self.len >= Self::MAX_CHANGES {
            return Err(DeltaError::TooManyChanges);
        }

        self.changes[self.len] = change;
        self.len += 1;
        Ok(())
    }

    pub fn push_change(
        &mut self,
        from: Square,
        to: Square,
        piece_from: Piece,
        piece_to: Piece,
    ) -> Result<(), DeltaError> {
        self.push(DeltaChange::new(from, to, piece_from, piece_to))
    }

    pub fn push_move(
        &mut self,
        from: Square,
        to: Square,
        piece_from: Piece,
        piece_to: Piece,
    ) -> Result<(), DeltaError> {
        self.push(DeltaChange::move_piece(from, to, piece_from, piece_to))
    }

    pub fn push_removal(&mut self, square: Square, piece: Piece) -> Result<(), DeltaError> {
        self.push(DeltaChange::removal(square, piece))
    }

    pub fn push_addition(&mut self, square: Square, piece: Piece) -> Result<(), DeltaError> {
        self.push(DeltaChange::addition(square, piece))
    }

    pub fn changes(&self) -> &[DeltaChange] {
        &self.changes[..self.len]
    }

    fn to_dirty_piece(self) -> DirtyPiece {
        let mut dirty = DirtyPiece::new();
        for change in self.changes() {
            dirty.add_change(
                change.from,
                change.to,
                change.piece_from.index(),
                change.piece_to.index(),
            );
        }
        dirty
    }
}

pub struct NNUEProbe {
    networks: Arc<NnueNetworks>,
    scratch_big: ScratchBuffer,
    scratch_small: ScratchBuffer,
    pieces: [Piece; 64],
    king_squares: [Square; 2],
    piece_count: usize,
    pawn_count: [i32; 2],
    non_pawn_material: [i32; 2],
    by_color_bb: [u64; 2],
    by_type_bb: [u64; 6],
    accumulator_stack: AccumulatorStack,
    finny_tables: FinnyTables,
}

impl NNUEProbe {
    pub fn new(big_path: &str, small_path: &str) -> io::Result<Self> {
        let networks = Arc::new(NnueNetworks::new(big_path, small_path)?);
        Ok(Self::from_networks(networks))
    }

    pub fn from_networks(networks: Arc<NnueNetworks>) -> Self {
        let scratch_big = ScratchBuffer::new(networks.big_net.feature_transformer.half_dims);
        let scratch_small = ScratchBuffer::new(networks.small_net.feature_transformer.half_dims);

        let mut finny_tables = FinnyTables::new();

        // Initialize with biases
        finny_tables.clear(
            &networks.big_net.feature_transformer.biases,
            &networks.small_net.feature_transformer.biases,
        );

        Self {
            networks,
            scratch_big,
            scratch_small,
            pieces: [Piece::None; 64],
            king_squares: [0; 2], // Default
            piece_count: 0,
            pawn_count: [0; 2],
            non_pawn_material: [0; 2],
            by_color_bb: [0; 2],
            by_type_bb: [0; 6],
            accumulator_stack: AccumulatorStack::new(),
            finny_tables,
        }
    }

    pub fn with_networks(networks: Arc<NnueNetworks>) -> io::Result<Self> {
        Ok(Self::from_networks(networks))
    }

    /// Set the root position - this does a full refresh
    pub fn set_position(&mut self, pieces: &[(Piece, Square)], rule50: i32) {
        // Reset state
        self.pieces = [Piece::None; 64];
        self.piece_count = 0;
        self.pawn_count = [0; 2];
        self.non_pawn_material = [0; 2];
        self.king_squares = [0; 2];
        self.by_color_bb = [0; 2];
        self.by_type_bb = [0; 6];

        // Note: We DO NOT clear Finny Tables here!
        // Stockfish persists the cache across positions.
        // Clearing it would force a full refresh on every node, killing performance.
        // The cache will correct itself lazily when a king lands on a square.

        for &(piece, square) in pieces {
            self.add_piece_internal(piece, square);
        }

        self.accumulator_stack.reset_with_refresh(
            self.king_squares,
            &self.networks.big_net.feature_transformer,
            &self.networks.small_net.feature_transformer,
            &mut self.finny_tables,
            self.by_color_bb,
            self.by_type_bb,
            rule50,
        );
    }

    /// Pre-populate Finny Tables with full accumulators for all 64 king squares
    /// Call this after set_position for maximum cache efficiency on king moves
    pub fn prepopulate_cache(&mut self) {
        let mut pieces_idx = PieceList::new();
        collect_pieces_from(&self.pieces, &mut pieces_idx);

        self.finny_tables.prepopulate(
            pieces_idx.as_slice(),
            &self.networks.big_net.feature_transformer,
            &self.networks.small_net.feature_transformer,
            self.king_squares,
        );
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
            let pt = piece.piece_type();
            if pt > 0 {
                let mask = 1u64 << square;
                self.by_color_bb[color.index()] |= mask;
                self.by_type_bb[pt - 1] |= mask;
            }
        }

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
            let pt = piece.piece_type();
            if pt > 0 {
                let mask = !(1u64 << square);
                self.by_color_bb[color.index()] &= mask;
                self.by_type_bb[pt - 1] &= mask;
            }
        }

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

    #[inline(always)]
    pub fn rule50(&self) -> i32 {
        self.accumulator_stack.latest().rule50
    }

    #[inline(always)]
    fn apply_delta_internal(&mut self, delta: MoveDelta) {
        for change in delta.changes() {
            if change.piece_from != Piece::None {
                debug_assert_eq!(self.pieces[change.from], change.piece_from);
                self.remove_piece_internal(change.from);
            }
        }

        for change in delta.changes() {
            if change.piece_to != Piece::None {
                debug_assert_eq!(self.pieces[change.to], Piece::None);
                self.add_piece_internal(change.piece_to, change.to);
            }
        }

        let dirty = delta.to_dirty_piece();
        self.accumulator_stack.push(&dirty, delta.next_rule50());

        let color_bb = self.by_color_bb;
        let type_bb = self.by_type_bb;
        self.accumulator_stack.update_incremental(
            self.king_squares,
            &self.networks.big_net.feature_transformer,
            &self.networks.small_net.feature_transformer,
            &mut self.finny_tables,
            || (color_bb, type_bb),
        );
    }

    #[inline(always)]
    fn undo_delta_internal(&mut self, delta: MoveDelta) {
        for change in delta.changes().iter().rev() {
            if change.piece_to != Piece::None {
                debug_assert_eq!(self.pieces[change.to], change.piece_to);
                self.remove_piece_internal(change.to);
            }
        }

        for change in delta.changes().iter().rev() {
            if change.piece_from != Piece::None {
                debug_assert_eq!(self.pieces[change.from], Piece::None);
                self.add_piece_internal(change.piece_from, change.from);
            }
        }

        self.accumulator_stack.pop();
    }

    #[inline(always)]
    pub fn apply_delta(&mut self, delta: MoveDelta) {
        self.apply_delta_internal(delta);
    }

    #[inline(always)]
    pub fn undo_delta(&mut self, delta: MoveDelta) {
        self.undo_delta_internal(delta);
    }

    #[inline(always)]
    pub fn make_null_move(&mut self) {
        self.apply_delta_internal(MoveDelta::null(self.rule50() + 1));
    }

    #[inline(always)]
    pub fn make_null_move_with_rule50(&mut self, next_rule50: i32) {
        self.apply_delta_internal(MoveDelta::null(next_rule50));
    }

    #[inline(always)]
    pub fn unmake_null_move(&mut self) {
        self.accumulator_stack.pop();
    }

    /// Make a move - pushes new state onto accumulator stack
    pub fn make_move(&mut self, from_sq: Square, to_sq: Square, piece: Piece) {
        let mut dirty = DirtyPiece::new();
        let from_piece = self.pieces[from_sq];
        let to_piece = self.pieces[to_sq];

        self.remove_piece_internal(from_sq);
        self.add_piece_internal(piece, to_sq);

        dirty.add_change(from_sq, to_sq, from_piece.index(), piece.index());

        if to_piece != Piece::None {
            dirty.add_change(to_sq, to_sq, to_piece.index(), Piece::None.index());
        }

        // Rule50 update (simple logic: reset on pawn move or capture, else increment)
        let prev_rule50 = self.rule50();
        let new_rule50 = if from_piece.piece_type() == 1 || to_piece != Piece::None {
            0
        } else {
            prev_rule50 + 1
        };

        self.accumulator_stack.push(&dirty, new_rule50);

        let color_bb = self.by_color_bb;
        let type_bb = self.by_type_bb;
        self.accumulator_stack.update_incremental(
            self.king_squares,
            &self.networks.big_net.feature_transformer,
            &self.networks.small_net.feature_transformer,
            &mut self.finny_tables,
            || (color_bb, type_bb),
        );
    }

    /// Unmake a move - pops state from accumulator stack (O(1)!)
    pub fn unmake_move(
        &mut self,
        from_sq: Square,
        to_sq: Square,
        from_piece: Piece,
        captured_piece: Option<Piece>,
    ) {
        self.remove_piece_internal(to_sq);

        if let Some(captured) = captured_piece {
            self.add_piece_internal(captured, to_sq);
        }

        self.add_piece_internal(from_piece, from_sq);
        self.accumulator_stack.pop();
    }

    /// Legacy update method - applies changes directly to current accumulators
    /// Does NOT use the stack - for one-off evaluations only
    pub fn update(&mut self, removed: &[(Piece, Square)], added: &[(Piece, Square)]) {
        if removed.len() > PIECE_LIST_CAPACITY || added.len() > PIECE_LIST_CAPACITY {
            let mut removed_mapped: Vec<(usize, usize)> = Vec::with_capacity(removed.len());
            let mut added_mapped: Vec<(usize, usize)> = Vec::with_capacity(added.len());

            let mut king_moved = false;

            for &(piece, square) in removed {
                self.remove_piece_internal(square);
                removed_mapped.push((square, piece.index()));
                if piece.is_king() {
                    king_moved = true;
                }
            }

            for &(piece, square) in added {
                self.add_piece_internal(piece, square);
                added_mapped.push((square, piece.index()));
                if piece.is_king() {
                    king_moved = true;
                }
            }

            if king_moved {
                self.refresh_accumulators();
            } else {
                let state = self.accumulator_stack.mut_latest();

                state.acc_big.update_with_ksq(
                    &added_mapped,
                    &removed_mapped,
                    self.king_squares,
                    &self.networks.big_net.feature_transformer,
                );
                state.acc_small.update_with_ksq(
                    &added_mapped,
                    &removed_mapped,
                    self.king_squares,
                    &self.networks.small_net.feature_transformer,
                );
            }
            return;
        }

        let mut removed_mapped = PieceList::new();
        let mut added_mapped = PieceList::new();

        // Track if king moved
        let mut king_moved = false;

        // Apply removals
        for &(piece, square) in removed {
            self.remove_piece_internal(square);
            removed_mapped.push(square, piece.index());
            if piece.is_king() {
                king_moved = true;
            }
        }

        // Apply additions
        for &(piece, square) in added {
            self.add_piece_internal(piece, square);
            added_mapped.push(square, piece.index());
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
                added_mapped.as_slice(),
                removed_mapped.as_slice(),
                self.king_squares,
                &self.networks.big_net.feature_transformer,
            );
            state.acc_small.update_with_ksq(
                added_mapped.as_slice(),
                removed_mapped.as_slice(),
                self.king_squares,
                &self.networks.small_net.feature_transformer,
            );
        }
    }

    fn refresh_accumulators(&mut self) {
        // Collect all pieces
        let mut pieces_idx = PieceList::new();
        collect_pieces_from(&self.pieces, &mut pieces_idx);

        self.accumulator_stack.refresh(
            pieces_idx.as_slice(),
            self.king_squares,
            &self.networks.big_net.feature_transformer,
            &self.networks.small_net.feature_transformer,
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
            let (psqt, pos) = self.networks.small_net.evaluate(
                &latest_state.acc_small,
                bucket,
                stm,
                &mut self.scratch_small,
            );
            nnue_val = (125 * psqt + 131 * pos) / 128;
            psqt_val = psqt;
            positional_val = pos;

            if nnue_val.abs() < 236 {
                // Use big network
                let (psqt_b, pos_b) = self.networks.big_net.evaluate(
                    &latest_state.acc_big,
                    bucket,
                    stm,
                    &mut self.scratch_big,
                );
                nnue_val = (125 * psqt_b + 131 * pos_b) / 128;
                psqt_val = psqt_b;
                positional_val = pos_b;
            }
        } else {
            // Use big network
            let (psqt, pos) = self.networks.big_net.evaluate(
                &latest_state.acc_big,
                bucket,
                stm,
                &mut self.scratch_big,
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
        let mut v = (nnue_val * (77777 + material) + optimism * (7777 + material)) / 77777;

        // Damp down the evaluation linearly when shuffling
        v -= v * latest_state.rule50 / 212;

        // Clamp to avoid tablebase range overlaps
        v.clamp(-31753, 31753)
    }
}
