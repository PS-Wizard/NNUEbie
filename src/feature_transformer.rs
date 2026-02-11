use crate::loader::{read_leb128_i16, read_leb128_i16_checked, read_leb128_i32};
use std::io::{self, Read};

pub const PSQT_BUCKETS: usize = 8;

pub struct FeatureTransformer {
    pub input_dims: usize,
    pub half_dims: usize,
    pub biases: Vec<i16>,
    pub weights: Vec<i16>,
    pub psqt_weights: Vec<i32>,
}

impl FeatureTransformer {
    pub fn new(input_dims: usize, half_dims: usize) -> Self {
        Self {
            input_dims,
            half_dims,
            biases: vec![],
            weights: vec![],
            psqt_weights: vec![],
        }
    }

    pub fn read_parameters<R: Read>(
        &mut self,
        reader: &mut R,
        skip_first_magic: bool,
    ) -> io::Result<()> {
        let biases = read_leb128_i16_checked(reader, self.half_dims, !skip_first_magic)?;
        let weights = read_leb128_i16(reader, self.half_dims * self.input_dims)?;
        let psqt_weights = read_leb128_i32(reader, PSQT_BUCKETS * self.input_dims)?;

        self.biases = biases;
        self.weights = weights;
        self.psqt_weights = psqt_weights;

        // Scale weights and biases by 2 (as per C++ read_parameters)
        // No permutation needed if we stick to scalar logic, assuming file is in standard order.
        // C++ code: permute_weights(); scale_weights(true);
        // scale_weights(true) multiplies by 2.

        for b in &mut self.biases {
            *b = b.wrapping_mul(2);
        }
        for w in &mut self.weights {
            *w = w.wrapping_mul(2);
        }

        Ok(())
    }
}
