use crate::accumulator::Accumulator;
use crate::feature_transformer::FeatureTransformer;
use crate::layers::{AffineTransform, ClippedReLU, Layer, SqrClippedReLU};
use crate::{OUTPUT_SCALE, WEIGHT_SCALE_BITS};
use std::fs::File;
use std::io::{self, BufReader, Read};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const LAYER_STACKS: usize = 8;

pub struct Network {
    pub feature_transformer: FeatureTransformer,
    pub fc_0: Vec<AffineTransform>,
    pub fc_1: Vec<AffineTransform>,
    pub fc_2: Vec<AffineTransform>,
    pub ac_sqr_0: SqrClippedReLU,
    pub ac_0: ClippedReLU,
    pub ac_1: ClippedReLU,
    pub is_big: bool,
}

pub struct ScratchBuffer {
    pub transformed_features: Vec<u8>,
    pub fc_0_out: Vec<i32>,
    pub ac_0_out: Vec<u8>, // Reused for ac_sqr_0 as well
    pub fc_1_in: Vec<u8>,
    pub fc_1_out: Vec<i32>,
    pub ac_1_out: Vec<u8>,
    pub fc_2_out: Vec<i32>,
}

impl ScratchBuffer {
    pub fn new(half_dims: usize) -> Self {
        Self {
            transformed_features: vec![0u8; half_dims],
            fc_0_out: vec![0i32; 16], // L2 + 1
            ac_0_out: vec![0u8; 16],  // L2 + 1
            fc_1_in: vec![0u8; 30],   // L2 * 2
            fc_1_out: vec![0i32; 32], // L3
            ac_1_out: vec![0u8; 32],  // L3
            fc_2_out: vec![0i32; 1],  // 1
        }
    }
}

impl Network {
    pub fn load(path: &str, is_big: bool) -> io::Result<Self> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        // Read Header
        let version = crate::loader::read_little_endian_u32(&mut reader)?;
        // println!("Version: {:x}", version);
        if version != crate::VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid version: {:x}, expected {:x}",
                    version,
                    crate::VERSION
                ),
            ));
        }
        let _hash = crate::loader::read_little_endian_u32(&mut reader)?;
        // println!("Network Hash: {:x}", _hash);

        let desc_len = crate::loader::read_little_endian_u32(&mut reader)? as usize;
        // println!("Desc Len: {}", desc_len);

        let mut desc = vec![0u8; desc_len];
        reader.read_exact(&mut desc)?;
        // println!("Desc: {:?}", String::from_utf8_lossy(&desc));

        // Feature Transformer
        let _hash_ft = crate::loader::read_little_endian_u32(&mut reader)?;
        // println!("FT Hash: {:x}", _hash_ft);

        // Peek/Consume Magic String (FeatureTransformer bias read doesn't expect it)
        let mut check = [0u8; 17];
        reader.read_exact(&mut check)?;
        if &check != b"COMPRESSED_LEB128" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid LEB128 magic string: {:?} (Ascii: {})",
                    check,
                    String::from_utf8_lossy(&check)
                ),
            ));
        }

        // Determine dims
        let (input_dims, half_dims, l2, l3) = if is_big {
            (22528, 3072, 15, 32)
        } else {
            (22528, 128, 15, 32)
        };

        let mut ft = FeatureTransformer::new(input_dims, half_dims);
        ft.read_parameters(&mut reader, true)?; // Skip first magic

        // Layers
        let mut fc_0s = Vec::with_capacity(LAYER_STACKS);
        let mut fc_1s = Vec::with_capacity(LAYER_STACKS);
        let mut fc_2s = Vec::with_capacity(LAYER_STACKS);

        for _ in 0..LAYER_STACKS {
            let _hash_stack = crate::loader::read_little_endian_u32(&mut reader)?;

            // fc_0
            let _fc_0 = AffineTransform::new(half_dims, l2);
            // fc_0 output is l2 + 1 (16) to include residual scaling factor
            let mut fc_0_layer = AffineTransform::new(half_dims, l2 + 1);
            fc_0_layer.read_parameters(&mut reader)?;

            // ac_sqr_0 (no params)
            // ac_0 (no params)

            // fc_1
            // fc_1 input is 30 (FC_0_OUTPUTS * 2), output is 32 (l3)
            let mut fc_1_layer = AffineTransform::new(l2 * 2, l3);
            fc_1_layer.read_parameters(&mut reader)?;

            // ac_1 (no params)

            // fc_2
            // Layers::AffineTransform<FC_1_OUTPUTS, 1>
            // Input 32, Output 1.
            let mut fc_2_layer = AffineTransform::new(l3, 1);
            fc_2_layer.read_parameters(&mut reader)?;

            fc_0s.push(fc_0_layer);
            fc_1s.push(fc_1_layer);
            fc_2s.push(fc_2_layer);
        }

        Ok(Self {
            feature_transformer: ft,
            fc_0: fc_0s,
            fc_1: fc_1s,
            fc_2: fc_2s,
            ac_sqr_0: SqrClippedReLU::new(l2 + 1),
            // Architecture:
            // ac_sqr_0.propagate(buffer.fc_0_out, buffer.ac_sqr_0_out);
            // ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
            // Input to both is fc_0 output (size 16).
            // But ac_sqr_0 is defined as <FC_0_OUTPUTS + 1>.
            // ac_0 is defined as <FC_0_OUTPUTS + 1>.
            // So both process 16 elements.
            ac_0: ClippedReLU::new(l2 + 1),
            ac_1: ClippedReLU::new(l3),
            is_big,
        })
    }

    fn transform_features(
        &self,
        accumulator: &Accumulator,
        scratch: &mut ScratchBuffer,
        us: usize,
        them: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return self.transform_features_avx2(accumulator, scratch, us, them);
            }
        }

        let half_dims = self.feature_transformer.half_dims;

        // Output filled: first half Us, second half Them.
        for p in 0..2 {
            let perspective = if p == 0 { us } else { them };
            let offset = (half_dims / 2) * p;

            for j in 0..(half_dims / 2) {
                let sum0 = accumulator.accumulation[perspective][j].clamp(0, 127 * 2) as i32;
                let sum1 = accumulator.accumulation[perspective][j + half_dims / 2]
                    .clamp(0, 127 * 2) as i32;

                scratch.transformed_features[offset + j] = ((sum0 * sum1) / 512) as u8;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn transform_features_avx2(
        &self,
        accumulator: &Accumulator,
        scratch: &mut ScratchBuffer,
        us: usize,
        them: usize,
    ) {
        let half_dims = self.feature_transformer.half_dims;
        let output_ptr = scratch.transformed_features.as_mut_ptr();

        for p in 0..2 {
            let perspective = if p == 0 { us } else { them };
            let offset = (half_dims / 2) * p;
            let acc_ptr = accumulator.accumulation[perspective].as_ptr();

            let chunk_size = 16;
            let n = (half_dims / 2) / chunk_size * chunk_size;

            for j in (0..n).step_by(chunk_size) {
                let v0 = _mm256_loadu_si256(acc_ptr.add(j) as *const _);
                let v1 = _mm256_loadu_si256(acc_ptr.add(j + half_dims / 2) as *const _);

                let min = _mm256_set1_epi16(254);
                let max = _mm256_setzero_si256();

                let v0_c = _mm256_max_epi16(max, _mm256_min_epi16(min, v0));
                let v1_c = _mm256_max_epi16(max, _mm256_min_epi16(min, v1));

                let prod = _mm256_mullo_epi16(v0_c, v1_c);
                let res = _mm256_srli_epi16(prod, 9);

                let lo = _mm256_castsi256_si128(res);
                let hi = _mm256_extracti128_si256(res, 1);

                let packed = _mm_packus_epi16(lo, hi);

                _mm_storeu_si128(output_ptr.add(offset + j) as *mut _, packed);
            }

            for j in n..(half_dims / 2) {
                let sum0 = accumulator.accumulation[perspective][j].clamp(0, 127 * 2) as i32;
                let sum1 = accumulator.accumulation[perspective][j + half_dims / 2]
                    .clamp(0, 127 * 2) as i32;
                *output_ptr.add(offset + j) = ((sum0 * sum1) / 512) as u8;
            }
        }
    }

    pub fn evaluate(
        &self,
        accumulator: &Accumulator,
        bucket: usize,
        side_to_move: usize,
        scratch: &mut ScratchBuffer,
    ) -> (i32, i32) {
        // 1. Transform Feature (Acc -> Output)
        // FeatureTransformer transform logic in C++:
        /*
          accumulatorStack.evaluate(pos, *this, *cache);
          ...
          psqt = (psqtAcc[us][bucket] - psqtAcc[them][bucket]) / 2;
          ...
          transform acc -> output
        */

        let us = side_to_move;
        let them = 1 - us;

        let psqt = (accumulator.psqt_accumulation[us][bucket]
            - accumulator.psqt_accumulation[them][bucket])
            / 2;

        self.transform_features(accumulator, scratch, us, them);

        // 2. Propagate MLP
        let _l2 = 15;
        // let l3 = 32;

        let fc_0 = &self.fc_0[bucket];
        let fc_1 = &self.fc_1[bucket];
        let fc_2 = &self.fc_2[bucket];

        fc_0.propagate(&scratch.transformed_features, &mut scratch.fc_0_out);

        self.ac_sqr_0
            .propagate(&scratch.fc_0_out, &mut scratch.ac_0_out);
        self.ac_0
            .propagate(&scratch.fc_0_out, &mut scratch.ac_0_out);

        // Concatenate for fc_1 input: [ac_sqr_0(0..15), ac_0(0..15)]
        // Overlap: ac_0[0] overwrites ac_sqr_0[15].
        // fc_1 input size 30: reads [0..29] effectively skipping the residual indices.
        // Copy first 15 from ac_sqr_0 (reuse ac_0_out buffer for first 15, but ac_0_out IS ac_0_out. Wait.)
        // ac_0_out now contains ac_0 output.
        // We need ac_sqr_0 output AND ac_0 output.
        // My scratch buffer only has one `ac_0_out`.
        // So I need to run ac_sqr_0 first, copy it, then run ac_0.

        // Strategy:
        // 1. Run ac_sqr_0 -> scratch.ac_0_out
        // 2. Copy scratch.ac_0_out[0..15] to scratch.fc_1_in[0..15]
        // 3. Run ac_0 -> scratch.ac_0_out (overwrites)
        // 4. Copy scratch.ac_0_out[0..15] to scratch.fc_1_in[15..30]

        self.ac_sqr_0
            .propagate(&scratch.fc_0_out, &mut scratch.ac_0_out);
        scratch.fc_1_in[0..15].copy_from_slice(&scratch.ac_0_out[0..15]);

        self.ac_0
            .propagate(&scratch.fc_0_out, &mut scratch.ac_0_out);
        scratch.fc_1_in[15..30].copy_from_slice(&scratch.ac_0_out[0..15]);

        fc_1.propagate(&scratch.fc_1_in, &mut scratch.fc_1_out);

        self.ac_1
            .propagate(&scratch.fc_1_out, &mut scratch.ac_1_out);

        fc_2.propagate(&scratch.ac_1_out, &mut scratch.fc_2_out);

        // Residual scaling
        let residual = scratch.fc_0_out[15];
        let fwd_out = residual * (600 * OUTPUT_SCALE) / (127 * (1 << WEIGHT_SCALE_BITS));

        let positional = scratch.fc_2_out[0] + fwd_out;

        // Result scaled by OutputScale
        (psqt / OUTPUT_SCALE, positional / OUTPUT_SCALE)
    }
}
