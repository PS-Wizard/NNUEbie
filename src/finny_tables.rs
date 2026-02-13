use crate::accumulator::Accumulator;
use crate::feature_transformer::FeatureTransformer;

const PSQT_BUCKETS: usize = 8;

/// Cache entry for a specific king square and color (Finny Table entry)
/// When the king is on this square, we can use this cached accumulator state
/// and apply incremental updates for pieces that differ.
pub struct AccumulatorCacheEntry<const SIZE: usize> {
    /// Cached accumulator values (biases + applied weights) - heap allocated to avoid stack overflow
    pub accumulation: Box<[i16; SIZE]>,
    /// Cached PSQT values
    pub psqt_accumulation: [i32; PSQT_BUCKETS],
    /// Bitboard of white pieces (0 if not cached)
    pub white_pieces: u64,
    /// Bitboard of black pieces (0 if not cached)
    pub black_pieces: u64,
    /// Whether this cache entry is valid
    pub valid: bool,
}

impl<const SIZE: usize> AccumulatorCacheEntry<SIZE> {
    pub fn new() -> Self {
        Self {
            accumulation: Box::new([0; SIZE]),
            psqt_accumulation: [0; PSQT_BUCKETS],
            white_pieces: 0,
            black_pieces: 0,
            valid: false,
        }
    }

    /// Initialize entry with just the biases (empty board state)
    pub fn clear(&mut self, biases: &[i16]) {
        self.accumulation.copy_from_slice(biases);
        self.psqt_accumulation.fill(0);
        self.white_pieces = 0;
        self.black_pieces = 0;
        self.valid = true;
    }

    /// Check if this entry can be used for the given position
    /// Returns the number of pieces that differ (0 means exact match)
    pub fn can_use(&self, white_pieces: u64, black_pieces: u64) -> Option<usize> {
        if !self.valid {
            return None;
        }

        // Count differing pieces using XOR
        let white_diff = self.white_pieces ^ white_pieces;
        let black_diff = self.black_pieces ^ black_pieces;

        // Count bits set (number of differing squares)
        let diff_count = white_diff.count_ones() + black_diff.count_ones();

        // If too many differences, it's not worth using the cache
        // Stockfish typically allows up to 3-4 pieces to differ
        if diff_count <= 4 {
            Some(diff_count as usize)
        } else {
            None
        }
    }
}

impl<const SIZE: usize> Default for AccumulatorCacheEntry<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Finny Table / Accumulator Cache for one network size
/// Contains 64 squares × 2 colors = 128 cache entries
pub struct AccumulatorCache<const SIZE: usize> {
    /// entries[square][color] - cached state for when king is on that square
    pub entries: [[AccumulatorCacheEntry<SIZE>; 2]; 64],
}

impl<const SIZE: usize> AccumulatorCache<SIZE> {
    pub fn new() -> Self {
        Self {
            entries: std::array::from_fn(|_| std::array::from_fn(|_| AccumulatorCacheEntry::new())),
        }
    }

    /// Initialize all entries with biases (called on new game or clear)
    pub fn clear(&mut self, biases: &[i16]) {
        for square_entries in &mut self.entries {
            for entry in square_entries {
                entry.clear(biases);
            }
        }
    }

    /// Try to refresh accumulator using cache
    /// Returns true if successful (used cache), false if full refresh needed
    pub fn try_refresh(
        &mut self,
        accumulator: &mut Accumulator<SIZE>,
        pieces: &[(usize, usize)], // (Square, Piece)
        king_squares: [usize; 2],
        ft: &FeatureTransformer,
    ) -> bool {
        let mut used_cache = [false; 2];

        for perspective in 0..2 {
            let king_sq = king_squares[perspective];
            let entry = &mut self.entries[king_sq][perspective];

            // Build bitboards for current position
            let mut white_bb: u64 = 0;
            let mut black_bb: u64 = 0;

            for &(sq, pc) in pieces {
                let color = pc / 8; // 0 = white, 1 = black
                if color == 0 {
                    white_bb |= 1u64 << sq;
                } else {
                    black_bb |= 1u64 << sq;
                }
            }

            // Check if we can use this cache entry (exact match only for now)
            if let Some(diff_count) = entry.can_use(white_bb, black_bb) {
                if diff_count == 0 {
                    // Exact match! Just copy from cache - this is the fast path
                    accumulator.accumulation[perspective].copy_from_slice(&*entry.accumulation);
                    accumulator.psqt_accumulation[perspective]
                        .copy_from_slice(&entry.psqt_accumulation);
                    used_cache[perspective] = true;
                }
                // Note: Partial matches (diff_count > 0) would require complex delta updates
                // For now, we fall back to full refresh when there's no exact match
                // This still gives significant speedup for common cases like:
                // - King returning to a previously visited square
                // - Transpositions in the search tree
            }

            if !used_cache[perspective] {
                // Full refresh needed for this perspective
                return false;
            }
        }

        true
    }

    /// Update cache entry after a full refresh
    pub fn update_cache(
        &mut self,
        accumulator: &Accumulator<SIZE>,
        pieces: &[(usize, usize)],
        king_squares: [usize; 2],
    ) {
        for perspective in 0..2 {
            let king_sq = king_squares[perspective];
            let entry = &mut self.entries[king_sq][perspective];

            // Store accumulator state
            entry
                .accumulation
                .copy_from_slice(&*accumulator.accumulation[perspective]);
            entry
                .psqt_accumulation
                .copy_from_slice(&accumulator.psqt_accumulation[perspective]);

            // Store piece bitboards
            entry.white_pieces = 0;
            entry.black_pieces = 0;

            for &(sq, pc) in pieces {
                let color = pc / 8;
                if color == 0 {
                    entry.white_pieces |= 1u64 << sq;
                } else {
                    entry.black_pieces |= 1u64 << sq;
                }
            }

            entry.valid = true;
        }
    }
}

impl<const SIZE: usize> Default for AccumulatorCache<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined caches for both big and small networks
pub struct FinnyTables {
    pub cache_big: AccumulatorCache<3072>,
    pub cache_small: AccumulatorCache<128>,
}

impl FinnyTables {
    pub fn new() -> Self {
        Self {
            cache_big: AccumulatorCache::new(),
            cache_small: AccumulatorCache::new(),
        }
    }

    pub fn clear(&mut self, biases_big: &[i16], biases_small: &[i16]) {
        self.cache_big.clear(biases_big);
        self.cache_small.clear(biases_small);
    }
}

impl Default for FinnyTables {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_entry_new() {
        let entry: AccumulatorCacheEntry<128> = AccumulatorCacheEntry::new();
        assert!(!entry.valid);
    }

    #[test]
    fn test_cache_entry_clear() {
        let mut entry: AccumulatorCacheEntry<128> = AccumulatorCacheEntry::new();
        let biases = vec![1i16; 128];
        entry.clear(&biases);
        assert!(entry.valid);
        assert_eq!(entry.accumulation[0], 1);
    }

    #[test]
    fn test_cache_can_use() {
        let mut entry: AccumulatorCacheEntry<128> = AccumulatorCacheEntry::new();
        let biases = vec![0i16; 128];
        entry.clear(&biases);

        // Set cached position: white piece on e2 (square 12)
        entry.white_pieces = 1u64 << 12;

        // Exact match
        let result = entry.can_use(1u64 << 12, 0);
        assert_eq!(result, Some(0));

        // One piece differs
        let result = entry.can_use(1u64 << 12 | 1u64 << 13, 0);
        assert_eq!(result, Some(1));

        // Too many differences
        let result = entry.can_use(0xFFFFFFFF, 0);
        assert_eq!(result, None);
    }
}
