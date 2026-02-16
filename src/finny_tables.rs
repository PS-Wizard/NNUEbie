use crate::accumulator::Accumulator;
use crate::aligned::AlignedBuffer;
use crate::feature_transformer::FeatureTransformer;

const PSQT_BUCKETS: usize = 8;
const MAX_DIFF_PIECES: usize = 4;

#[derive(Clone, Copy)]
pub struct PieceInfo {
    square: usize,
    piece: usize,
}

pub struct AccumulatorCacheEntry<const SIZE: usize> {
    pub accumulation: AlignedBuffer<i16>,
    pub psqt_accumulation: [i32; PSQT_BUCKETS],
    pub by_color_bb: [u64; 2],
    pub by_type_bb: [u64; 6],
    pub valid: bool,
}

impl<const SIZE: usize> AccumulatorCacheEntry<SIZE> {
    pub fn new() -> Self {
        Self {
            accumulation: AlignedBuffer::new(SIZE),
            psqt_accumulation: [0; PSQT_BUCKETS],
            by_color_bb: [0; 2],
            by_type_bb: [0; 6],
            valid: false,
        }
    }

    pub fn clear(&mut self, biases: &[i16]) {
        self.accumulation.copy_from_slice(biases);
        self.psqt_accumulation.fill(0);
        self.by_color_bb.fill(0);
        self.by_type_bb.fill(0);
        self.valid = true;
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn compute_diff(
        &self,
        pieces: &[(usize, usize)],
        _king_sq: usize,
    ) -> Option<(Vec<PieceInfo>, Vec<PieceInfo>)> {
        if !self.valid {
            return None;
        }

        let mut current_color_bb = [0u64; 2];
        let mut current_type_bb = [0u64; 6];

        for &(sq, pc) in pieces {
            let color = pc / 8;
            let piece_type = match pc % 8 {
                1 => 0,
                2 => 1,
                3 => 2,
                4 => 3,
                5 => 4,
                6 => 5,
                _ => continue,
            };
            current_color_bb[color] |= 1u64 << sq;
            current_type_bb[piece_type] |= 1u64 << sq;
        }

        let mut removed = Vec::new();
        let mut added = Vec::new();

        for color in 0..2 {
            let old_color = self.by_color_bb[color];
            let new_color = current_color_bb[color];
            let diff = old_color ^ new_color;
            let to_remove = old_color & diff;
            let to_add = new_color & diff;

            for piece_type in 0..6 {
                let old_type = self.by_type_bb[piece_type];
                let new_type = current_type_bb[piece_type];

                let pieces_removed = to_remove & old_type;
                let pieces_added = to_add & new_type;

                let mut bb = pieces_removed;
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    removed.push(PieceInfo {
                        square: sq,
                        piece: color * 8 + piece_type + 1,
                    });
                    bb &= bb - 1;
                }

                let mut bb = pieces_added;
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    added.push(PieceInfo {
                        square: sq,
                        piece: color * 8 + piece_type + 1,
                    });
                    bb &= bb - 1;
                }
            }
        }

        let diff_count = removed.len() + added.len();

        if diff_count <= MAX_DIFF_PIECES {
            Some((removed, added))
        } else {
            None
        }
    }

    pub fn count_diff(&self, pieces: &[(usize, usize)]) -> Option<usize> {
        if !self.valid {
            return None;
        }

        let mut current_color_bb = [0u64; 2];

        for &(sq, pc) in pieces {
            let color = pc / 8;
            current_color_bb[color] |= 1u64 << sq;
        }

        let white_diff = self.by_color_bb[0] ^ current_color_bb[0];
        let black_diff = self.by_color_bb[1] ^ current_color_bb[1];

        let diff_count = (white_diff.count_ones() + black_diff.count_ones()) as usize;

        if diff_count <= MAX_DIFF_PIECES {
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

pub struct AccumulatorCache<const SIZE: usize> {
    pub entries: [[AccumulatorCacheEntry<SIZE>; 2]; 64],
}

impl<const SIZE: usize> AccumulatorCache<SIZE> {
    pub fn new() -> Self {
        Self {
            entries: std::array::from_fn(|_| std::array::from_fn(|_| AccumulatorCacheEntry::new())),
        }
    }

    pub fn clear(&mut self, biases: &[i16]) {
        for square_entries in &mut self.entries {
            for entry in square_entries {
                entry.clear(biases);
            }
        }
    }

    /// Pre-populate all 64 king square entries for a given position
    /// For each king square, compute the accumulator for that king position with all pieces
    pub fn prepopulate(
        &mut self,
        pieces: &[(usize, usize)],
        ft: &crate::feature_transformer::FeatureTransformer,
        _king_squares: [usize; 2],
    ) {
        for king_sq in 0..64 {
            // Compute accumulator for each king square
            let mut acc = crate::accumulator::Accumulator::<SIZE>::new();

            // Initialize with biases
            acc.accumulation[0].copy_from_slice(&ft.biases);
            acc.accumulation[1].copy_from_slice(&ft.biases);
            acc.psqt_accumulation[0].fill(0);
            acc.psqt_accumulation[1].fill(0);

            // Add features for this king square
            let ksq = [king_sq, king_sq]; // Both perspectives use same king square

            for &(sq, pc) in pieces {
                let idx_w = crate::features::make_index(
                    crate::features::WHITE,
                    sq,
                    pc,
                    ksq[crate::features::WHITE],
                );
                acc.add_feature(crate::features::WHITE, idx_w, ft);

                let idx_b = crate::features::make_index(
                    crate::features::BLACK,
                    sq,
                    pc,
                    ksq[crate::features::BLACK],
                );
                acc.add_feature(crate::features::BLACK, idx_b, ft);
            }

            // Store in cache for both perspectives
            for perspective in 0..2 {
                let entry = &mut self.entries[king_sq][perspective];
                entry
                    .accumulation
                    .copy_from_slice(&acc.accumulation[perspective]);
                entry.psqt_accumulation = acc.psqt_accumulation[perspective];
                entry.valid = true;

                // Compute bitboards for this entry
                entry.by_color_bb.fill(0);
                entry.by_type_bb.fill(0);

                for &(sq, pc) in pieces {
                    let color = pc / 8;
                    entry.by_color_bb[color] |= 1u64 << sq;

                    let piece_type = match pc % 8 {
                        1 => 0,
                        2 => 1,
                        3 => 2,
                        4 => 3,
                        5 => 4,
                        6 => 5,
                        _ => continue,
                    };
                    entry.by_type_bb[piece_type] |= 1u64 << sq;
                }
            }
        }
    }

    pub fn try_refresh(
        &mut self,
        accumulator: &mut Accumulator<SIZE>,
        pieces: &[(usize, usize)],
        king_squares: [usize; 2],
        ft: &FeatureTransformer,
    ) -> bool {
        for perspective in 0..2 {
            let king_sq = king_squares[perspective];
            let entry = &mut self.entries[king_sq][perspective];

            if !entry.is_valid() {
                return false;
            }

            if let Some((removed, added)) = entry.compute_diff(pieces, king_sq) {
                let removed_slice: Vec<(usize, usize)> =
                    removed.iter().map(|p| (p.square, p.piece)).collect();
                let added_slice: Vec<(usize, usize)> =
                    added.iter().map(|p| (p.square, p.piece)).collect();

                accumulator.accumulation[perspective].copy_from_slice(&*entry.accumulation);
                accumulator.psqt_accumulation[perspective]
                    .copy_from_slice(&entry.psqt_accumulation);

                accumulator.update_with_ksq(&added_slice, &removed_slice, king_squares, ft);
            } else {
                return false;
            }
        }

        true
    }

    pub fn update_cache(
        &mut self,
        accumulator: &Accumulator<SIZE>,
        pieces: &[(usize, usize)],
        king_squares: [usize; 2],
    ) {
        for perspective in 0..2 {
            let king_sq = king_squares[perspective];
            let entry = &mut self.entries[king_sq][perspective];

            entry
                .accumulation
                .copy_from_slice(accumulator.accumulation[perspective].as_slice());
            entry
                .psqt_accumulation
                .copy_from_slice(&accumulator.psqt_accumulation[perspective]);

            entry.by_color_bb.fill(0);
            entry.by_type_bb.fill(0);

            for &(sq, pc) in pieces {
                let color = pc / 8;
                entry.by_color_bb[color] |= 1u64 << sq;

                let piece_type = match pc % 8 {
                    1 => 0,
                    2 => 1,
                    3 => 2,
                    4 => 3,
                    5 => 4,
                    6 => 5,
                    _ => continue,
                };
                entry.by_type_bb[piece_type] |= 1u64 << sq;
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

    /// Pre-populate Finny Tables for all 64 king squares
    /// Computes full accumulators for each king square × 2 perspectives = 128 entries
    pub fn prepopulate(
        &mut self,
        pieces: &[(usize, usize)],
        ft_big: &crate::feature_transformer::FeatureTransformer,
        ft_small: &crate::feature_transformer::FeatureTransformer,
        king_squares: [usize; 2],
    ) {
        self.cache_big.prepopulate(pieces, ft_big, king_squares);
        self.cache_small.prepopulate(pieces, ft_small, king_squares);
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
    fn test_cache_diff_count() {
        let mut entry: AccumulatorCacheEntry<128> = AccumulatorCacheEntry::new();
        let biases = vec![0i16; 128];
        entry.clear(&biases);

        entry.by_color_bb[0] = 1u64 << 12;
        entry.by_type_bb[0] = 1u64 << 12;

        let pieces = vec![(12, 1usize)];
        let diff = entry.count_diff(&pieces);
        assert_eq!(diff, Some(0));

        let pieces = vec![(12, 1), (13, 1)];
        let diff = entry.count_diff(&pieces);
        assert_eq!(diff, Some(1));

        let pieces: Vec<(usize, usize)> = (0..32).map(|i| (i, 1)).collect();
        let diff = entry.count_diff(&pieces);
        assert_eq!(diff, None);
    }
}
