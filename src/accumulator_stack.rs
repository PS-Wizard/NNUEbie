use crate::accumulator::Accumulator;
use crate::feature_transformer::FeatureTransformer;

const MAX_PLY: usize = 128; // Maximum search depth + some margin

/// Information about what pieces changed between positions
#[derive(Clone, Debug)]
pub struct DirtyPiece {
    /// Number of pieces that changed
    pub dirty_num: usize,
    /// Squares where pieces were removed (from_sq for moves, captured squares)
    pub from: [usize; 3],
    /// Squares where pieces were added (to_sq for moves, promotion pieces)
    pub to: [usize; 3],
    /// Piece types that were removed
    pub piece_from: [usize; 3],
    /// Piece types that were added  
    pub piece_to: [usize; 3],
}

impl DirtyPiece {
    pub fn new() -> Self {
        Self {
            dirty_num: 0,
            from: [0; 3],
            to: [0; 3],
            piece_from: [0; 3],
            piece_to: [0; 3],
        }
    }

    pub fn reset(&mut self) {
        self.dirty_num = 0;
    }

    pub fn add_change(&mut self, from_sq: usize, to_sq: usize, piece_from: usize, piece_to: usize) {
        if self.dirty_num < 3 {
            self.from[self.dirty_num] = from_sq;
            self.to[self.dirty_num] = to_sq;
            self.piece_from[self.dirty_num] = piece_from;
            self.piece_to[self.dirty_num] = piece_to;
            self.dirty_num += 1;
        }
    }
}

impl Default for DirtyPiece {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulator state for a single position in the search tree
#[derive(Clone)]
pub struct AccumulatorState {
    pub acc_big: Accumulator<3072>,
    pub acc_small: Accumulator<128>,
    pub dirty_piece: DirtyPiece,
    pub computed: [bool; 2], // Track if each perspective is computed
}

impl AccumulatorState {
    pub fn new() -> Self {
        Self {
            acc_big: Accumulator::new(),
            acc_small: Accumulator::new(),
            dirty_piece: DirtyPiece::new(),
            computed: [false, false],
        }
    }

    pub fn reset(&mut self, dp: &DirtyPiece) {
        self.dirty_piece = dp.clone();
        self.computed = [false, false];
    }
}

impl Default for AccumulatorState {
    fn default() -> Self {
        Self::new()
    }
}

/// Stack of accumulator states for efficient make/unmake in search
pub struct AccumulatorStack {
    stack: Vec<AccumulatorState>,
    current_idx: usize,
}

impl AccumulatorStack {
    pub fn new() -> Self {
        let mut stack = Vec::with_capacity(MAX_PLY + 1);
        // Initialize with one empty state at index 0
        stack.push(AccumulatorState::new());

        Self {
            stack,
            current_idx: 1, // Start at 1 so we can always access current_idx - 1
        }
    }

    /// Get the latest (current) accumulator state
    pub fn latest(&self) -> &AccumulatorState {
        &self.stack[self.current_idx - 1]
    }

    /// Get mutable reference to latest state
    pub fn mut_latest(&mut self) -> &mut AccumulatorState {
        &mut self.stack[self.current_idx - 1]
    }

    /// Get the current stack index
    pub fn current_index(&self) -> usize {
        self.current_idx
    }

    /// Get mutable reference to state at specific index (for advanced use)
    pub fn state_at_mut(&mut self, idx: usize) -> &mut AccumulatorState {
        &mut self.stack[idx]
    }

    /// Push a new position onto the stack (make move)
    /// Stockfish-style: no deep copy of accumulators - just mark as not computed
    /// This avoids copying ~12KB on every move
    pub fn push(&mut self, dirty_piece: &DirtyPiece) {
        assert!(
            self.current_idx + 1 < self.stack.capacity(),
            "AccumulatorStack overflow - increase MAX_PLY"
        );

        // If we need to grow the stack
        if self.current_idx >= self.stack.len() {
            self.stack.push(AccumulatorState::new());
        }

        // Stockfish approach: NO copying!
        // Just mark the state as needing recomputation
        // The accumulators from previous state will be reused via incremental updates
        self.stack[self.current_idx].reset(dirty_piece);

        self.current_idx += 1;
    }

    /// Pop a position from the stack (unmake move) - O(1)!
    pub fn pop(&mut self) {
        assert!(self.current_idx > 1, "Cannot pop root position");
        self.current_idx -= 1;
    }

    /// Reset the stack to a single position (root position)
    pub fn reset(&mut self) {
        self.current_idx = 1;
        self.stack[0] = AccumulatorState::new();
    }

    /// Incrementally update accumulators from the previous state
    pub fn update_incremental(
        &mut self,
        king_squares: [usize; 2],
        ft_big: &FeatureTransformer,
        ft_small: &FeatureTransformer,
    ) {
        // Find the last state with computed accumulators BEFORE mutably borrowing
        // This is the Stockfish approach - lazily find the last usable state
        let current_idx = self.current_idx;
        let mut last_computed_idx = current_idx - 1;
        while last_computed_idx > 0 {
            if self.stack[last_computed_idx].computed == [true, true] {
                break;
            }
            last_computed_idx -= 1;
        }

        // If we found a computed state, copy its accumulator data to the current state
        // This is needed because we no longer clone on push()
        // Note: last_computed_idx can be 0 (root position is valid)
        let source_accum: Option<(
            Vec<i16>,
            Vec<i16>,
            [[i32; 8]; 2],
            Vec<i16>,
            Vec<i16>,
            [[i32; 8]; 2],
        )> = if self.stack[last_computed_idx].computed == [true, true] {
            let source = &self.stack[last_computed_idx];
            Some((
                source.acc_big.accumulation[0].as_slice().to_vec(),
                source.acc_big.accumulation[1].as_slice().to_vec(),
                source.acc_big.psqt_accumulation,
                source.acc_small.accumulation[0].as_slice().to_vec(),
                source.acc_small.accumulation[1].as_slice().to_vec(),
                source.acc_small.psqt_accumulation,
            ))
        } else {
            None
        };

        let current = self.mut_latest();

        // Copy accumulator data from source if we found one, otherwise initialize with biases
        if let Some((ab0, ab1, ab_psqt, as0, as1, as_psqt)) = source_accum {
            current.acc_big.accumulation[0].copy_from_slice(&ab0);
            current.acc_big.accumulation[1].copy_from_slice(&ab1);
            current.acc_big.psqt_accumulation.copy_from_slice(&ab_psqt);
            current.acc_small.accumulation[0].copy_from_slice(&as0);
            current.acc_small.accumulation[1].copy_from_slice(&as1);
            current
                .acc_small
                .psqt_accumulation
                .copy_from_slice(&as_psqt);
        } else {
            // Initialize with biases (full refresh case)
            current.acc_big.accumulation[0].copy_from_slice(&*ft_big.biases);
            current.acc_big.accumulation[1].copy_from_slice(&*ft_big.biases);
            current.acc_big.psqt_accumulation.fill([0; 8]);
            current.acc_small.accumulation[0].copy_from_slice(&*ft_small.biases);
            current.acc_small.accumulation[1].copy_from_slice(&*ft_small.biases);
            current.acc_small.psqt_accumulation.fill([0; 8]);
        }

        // Build change lists from dirty_piece
        let mut removed: Vec<(usize, usize)> = Vec::with_capacity(3);
        let mut added: Vec<(usize, usize)> = Vec::with_capacity(3);

        for i in 0..current.dirty_piece.dirty_num {
            // Piece moved from -> to
            removed.push((
                current.dirty_piece.from[i],
                current.dirty_piece.piece_from[i],
            ));
            added.push((current.dirty_piece.to[i], current.dirty_piece.piece_to[i]));
        }

        // Update both networks incrementally
        current
            .acc_big
            .update_with_ksq(&added, &removed, king_squares, ft_big);
        current
            .acc_small
            .update_with_ksq(&added, &removed, king_squares, ft_small);
        current.computed = [true, true];
    }

    /// Refresh accumulators from scratch (for root position or when needed)
    pub fn refresh(
        &mut self,
        pieces: &[(usize, usize)], // (Square, Piece)
        king_squares: [usize; 2],
        ft_big: &FeatureTransformer,
        ft_small: &FeatureTransformer,
    ) {
        let current = self.mut_latest();
        current.acc_big.refresh(pieces, king_squares, ft_big);
        current.acc_small.refresh(pieces, king_squares, ft_small);
        current.computed = [true, true];
    }
}

impl Default for AccumulatorStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_stack_new() {
        let stack = AccumulatorStack::new();
        assert_eq!(stack.current_idx, 1);
        assert_eq!(stack.stack.len(), 1);
    }

    #[test]
    fn test_accumulator_stack_push_pop() {
        let mut stack = AccumulatorStack::new();

        // Push a move
        let mut dp = DirtyPiece::new();
        dp.add_change(12, 28, 1, 1); // White pawn e2->e4
        stack.push(&dp);

        assert_eq!(stack.current_idx, 2);
        assert_eq!(stack.latest().dirty_piece.dirty_num, 1);
        assert_eq!(stack.latest().dirty_piece.from[0], 12);

        // Pop
        stack.pop();
        assert_eq!(stack.current_idx, 1);
    }

    #[test]
    fn test_accumulator_stack_multiple_pushes() {
        let mut stack = AccumulatorStack::new();

        // Simulate a sequence of moves
        for i in 0..10 {
            let mut dp = DirtyPiece::new();
            dp.add_change(i, i + 1, 1, 1);
            stack.push(&dp);
        }

        assert_eq!(stack.current_idx, 11);

        // Pop all
        for _ in 0..10 {
            stack.pop();
        }

        assert_eq!(stack.current_idx, 1);
    }
}
