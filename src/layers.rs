use crate::loader::{read_i32_array, read_i8_array};
use std::io::{self, Read};

pub trait Layer {
    type Input;
    type Output;
    fn propagate(&self, input: &[Self::Input], output: &mut [Self::Output]);
    fn read_parameters<R: Read>(&mut self, reader: &mut R) -> io::Result<()>;
}

pub struct AffineTransform {
    pub biases: Vec<i32>,
    pub weights: Vec<i8>,
    pub input_dims: usize,
    pub output_dims: usize,
    pub padded_input_dims: usize,
}

impl AffineTransform {
    pub fn new(input_dims: usize, output_dims: usize) -> Self {
        let padded_input_dims = (input_dims + 31) / 32 * 32;
        Self {
            biases: vec![0; output_dims],
            weights: vec![0; output_dims * padded_input_dims],
            input_dims,
            output_dims,
            padded_input_dims,
        }
    }
}

impl Layer for AffineTransform {
    type Input = u8;
    type Output = i32;

    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        // Fallback implementation (row-major weights)
        // input is effectively u8 but passed as u8 slices.
        // biases are i32

        output.copy_from_slice(&self.biases);

        // Standard dense matrix multiplication
        // weights are [output_dims][padded_input_dims] in memory (row-major)

        for i in 0..self.input_dims {
            let in_val = input[i];
            if in_val == 0 {
                continue;
            }
            let in_val_i32 = in_val as i32;

            for j in 0..self.output_dims {
                // weights[j][i]
                let weight_idx = j * self.padded_input_dims + i;
                let w = self.weights[weight_idx] as i32;
                output[j] += w * in_val_i32;
            }
        }
    }

    fn read_parameters<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
        let biases = read_i32_array(reader, self.output_dims)?;
        self.biases = biases;

        // C++ read_parameters: reads linearly from file -> writes to weights[get_weight_index(i)]
        // C++ fallback propagate: reads weights[j * Padded + i].
        // If get_weight_index(i) == i (fallback mode), then file order == Row-Major order.
        // Since we are replicating behavior, and we want to use Row-Major order for simplicity,
        // we can just read linearly.

        let weights_raw = read_i8_array(reader, self.output_dims * self.padded_input_dims)?;
        self.weights = weights_raw;

        Ok(())
    }
}

pub type AffineTransformSparseInput = AffineTransform;

pub struct ClippedReLU {
    pub dims: usize,
}

impl ClippedReLU {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Layer for ClippedReLU {
    type Input = i32;
    type Output = u8;

    fn propagate(&self, input: &[i32], output: &mut [u8]) {
        for (i, &val) in input.iter().enumerate().take(self.dims) {
            let scaled = val >> 6; // WeightScaleBits = 6
            output[i] = scaled.clamp(0, 127) as u8;
        }
    }

    fn read_parameters<R: Read>(&mut self, _reader: &mut R) -> io::Result<()> {
        Ok(())
    }
}

pub struct SqrClippedReLU {
    pub dims: usize,
}

impl SqrClippedReLU {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Layer for SqrClippedReLU {
    type Input = i32;
    type Output = u8;

    fn propagate(&self, input: &[i32], output: &mut [u8]) {
        for (i, &val) in input.iter().enumerate().take(self.dims) {
            let val_i64 = val as i64;
            let squared = val_i64 * val_i64;
            // >> (2 * 6 + 7) = 19
            let scaled = squared >> 19;
            output[i] = scaled.clamp(0, 127) as u8;
        }
    }

    fn read_parameters<R: Read>(&mut self, _reader: &mut R) -> io::Result<()> {
        Ok(())
    }
}
