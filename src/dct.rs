#![allow(dead_code)]

// adapted from https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms

// in the original these are arrays, but are only ever referred to with constant indices, so we can just unfold the whole array
// also storing reciprocals to replace divides with multiplies

const S_0: f32 = 0.353553390593273762200422;
const S_1: f32 = 0.254897789552079584470970;
const S_2: f32 = 0.270598050073098492199862;
const S_3: f32 = 0.300672443467522640271861;
const S_4: f32 = 0.353553390593273762200422;
const S_5: f32 = 0.449988111568207852319255;
const S_6: f32 = 0.653281482438188263928322;
const S_7: f32 = 1.281457723870753089398043;

const RS_0: f32 = 1.0 / S_0;
const RS_1: f32 = 1.0 / S_1;
const RS_2: f32 = 1.0 / S_2;
const RS_3: f32 = 1.0 / S_3;
const RS_4: f32 = 1.0 / S_4;
const RS_5: f32 = 1.0 / S_5;
const RS_6: f32 = 1.0 / S_6;
const RS_7: f32 = 1.0 / S_7;

const A_0: f32 = f32::NAN;
const A_1: f32 = 0.707106781186547524400844;
const A_2: f32 = 0.541196100146196984399723;
const A_3: f32 = 0.707106781186547524400844;
const A_4: f32 = 1.306562964876376527856643;
const A_5: f32 = 0.382683432365089771728460;

const RA_0: f32 = 1.0 / A_0;
const RA_1: f32 = 1.0 / A_1;
const RA_2: f32 = 1.0 / A_2;
const RA_3: f32 = 1.0 / A_3;
const RA_4: f32 = 1.0 / A_4;
const RA_5: f32 = 1.0 / A_5;

/// Quantization table for intra-frames (I-Frames)
pub static Q_TABLE_INTRA: [f32;64] = [
    8.0, 16.0, 19.0, 22.0, 26.0, 27.0, 29.0, 34.0,
    16.0, 16.0, 22.0, 24.0, 27.0, 29.0, 34.0, 37.0,
    19.0, 22.0, 26.0, 27.0, 29.0, 34.0, 34.0, 38.0,
    22.0, 22.0, 26.0, 27.0, 29.0, 34.0, 37.0, 40.0,
    22.0, 26.0, 27.0, 29.0, 32.0, 35.0, 40.0, 48.0,
    26.0, 27.0, 29.0, 32.0, 35.0, 40.0, 48.0, 58.0,
    26.0, 27.0, 29.0, 34.0, 38.0, 46.0, 56.0, 69.0,
    27.0, 29.0, 35.0, 38.0, 46.0, 56.0, 69.0, 83.0,
];

/// Quantization table for inter-frames (P-Frames)
pub static Q_TABLE_INTER: [f32;64] = [
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
];

pub static INV_ZIGZAG_TABLE: [usize;64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 
33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
];

pub static ZIGZAG_TABLE: [usize;64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22,
15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
];

/// Represents an 8x8 row-order matrix of DCT coefficients
#[derive(Clone, Copy, Debug)]
pub struct DctMatrix8x8 {
    pub m: [f32;64]
}

/// Represents an 8x8 row-order matrix of quantized DCT coefficients
#[derive(Clone, Copy, Debug)]
pub struct DctQuantizedMatrix8x8 {
    pub m: [i16;64]
}

impl DctQuantizedMatrix8x8 {
    pub fn from_slice(slice: &[i16]) -> DctQuantizedMatrix8x8 {
        let mut result = DctQuantizedMatrix8x8 { m: [0;64] };
        result.m.copy_from_slice(slice);

        result
    }
}

impl DctMatrix8x8 {
    pub fn new() -> DctMatrix8x8 {
        DctMatrix8x8 { m: [0.0;64] }
    }

    pub fn decode(src: &DctQuantizedMatrix8x8, q_table: &[f32;64]) -> DctMatrix8x8 {
        let mut result = DctMatrix8x8 { m: [0.0;64] };

        for idx in 0..64 {
            let c = src.m[INV_ZIGZAG_TABLE[idx]];
            let n = c as f32;
            let d = q_table[idx];

            result.m[idx] = n * d;
        }

        result
    }

    pub fn encode(self: &mut DctMatrix8x8, q_table: &[f32;64]) -> DctQuantizedMatrix8x8 {
        let mut result = DctQuantizedMatrix8x8 { m: [0;64] };

        for idx in 0..64 {
            let n = self.m[ZIGZAG_TABLE[idx]];
            let d = q_table[idx];

            result.m[idx] = (n / d) as i16;
        }

        result
    }

    pub fn get_row(self: &DctMatrix8x8, index: usize) -> [f32;8] {
        assert!(index < 8);
        let row_offset = index * 8;

        let mut result = [0.0;8];
        result.copy_from_slice(&self.m[row_offset..row_offset+8]);

        result
    }

    pub fn set_row(self: &mut DctMatrix8x8, index: usize, row: [f32;8]) {
        assert!(index < 8);
        let row_offset = index * 8;

        self.m[row_offset..row_offset+8].copy_from_slice(&row);
    }
    
    pub fn get_column(self: &DctMatrix8x8, index: usize) -> [f32;8] {
        assert!(index < 8);

        let mut result = [0.0;8];

        for row in 0..8 {
            result[row] = self.m[index + (row * 8)];
        }

        result
    }

    pub fn set_column(self: &mut DctMatrix8x8, index: usize, column: [f32;8]) {
        assert!(index < 8);

        for row in 0..8 {
            self.m[index + (row * 8)] = column[row];
        }
    }

    /// Perform an in-place DCT transformation of each row of this matrix
    pub fn dct_transform_rows(self: &mut DctMatrix8x8) {
        for idx in 0..8 {
            let mut row = self.get_row(idx);
            DctMatrix8x8::fast_dct8_transform(&mut row);
            self.set_row(idx, row);
        }
    }

    /// Perform an in-place DCT transformation of each column of this matrix
    pub fn dct_transform_columns(self: &mut DctMatrix8x8) {
        for idx in 0..8 {
            let mut column = self.get_column(idx);
            DctMatrix8x8::fast_dct8_transform(&mut column);
            self.set_column(idx, column);
        }
    }

    /// Perform an in-place inverse DCT transformation of each row of this matrix
    pub fn dct_inverse_transform_rows(self: &mut DctMatrix8x8) {
        for idx in 0..8 {
            let mut row = self.get_row(idx);
            DctMatrix8x8::fast_dct8_inverse_transform(&mut row);
            self.set_row(idx, row);
        }
    }

    /// Perform an in-place inverse DCT transformation of each column of this matrix
    pub fn dct_inverse_transform_columns(self: &mut DctMatrix8x8) {
        for idx in 0..8 {
            let mut column = self.get_column(idx);
            DctMatrix8x8::fast_dct8_inverse_transform(&mut column);
            self.set_column(idx, column);
        }
    }

    pub fn fast_dct8_transform(vector: &mut [f32;8]) {
        let v0 = vector[0] + vector[7];
        let v1 = vector[1] + vector[6];
        let v2 = vector[2] + vector[5];
        let v3 = vector[3] + vector[4];
        let v4 = vector[3] - vector[4];
        let v5 = vector[2] - vector[5];
        let v6 = vector[1] - vector[6];
        let v7 = vector[0] - vector[7];

        let v8 = v0 + v3;
        let v9 = v1 + v2;
        let v10 = v1 - v2;
        let v11 = v0 - v3;
        let v12 = -v4 - v5;
        let v13 = (v5 + v6) * A_3;
        let v14 = v6 + v7;

        let v15 = v8 + v9;
        let v16 = v8 - v9;
        let v17 = (v10 + v11) * A_1;
        let v18 = (v12 + v14) * A_5;

        let v19 = -v12 * A_2 - v18;
        let v20 = v14 * A_4 - v18;

        let v21 = v17 + v11;
        let v22 = v11 - v17;
        let v23 = v13 + v7;
        let v24 = v7 - v13;

        let v25 = v19 + v24;
        let v26 = v23 + v20;
        let v27 = v23 - v20;
        let v28 = v24 - v19;

        vector[0] = S_0 * v15;
        vector[1] = S_1 * v26;
        vector[2] = S_2 * v21;
        vector[3] = S_3 * v28;
        vector[4] = S_4 * v16;
        vector[5] = S_5 * v25;
        vector[6] = S_6 * v22;
        vector[7] = S_7 * v27;
    }

    pub fn fast_dct8_inverse_transform(vector: &mut [f32;8]) {
        let v15 = vector[0] * RS_0;
        let v26 = vector[1] * RS_1;
        let v21 = vector[2] * RS_2;
        let v28 = vector[3] * RS_3;
        let v16 = vector[4] * RS_4;
        let v25 = vector[5] * RS_5;
        let v22 = vector[6] * RS_6;
        let v27 = vector[7] * RS_7;
        
        let v19 = (v25 - v28) * 0.5;
        let v20 = (v26 - v27) * 0.5;
        let v23 = (v26 + v27) * 0.5;
        let v24 = (v25 + v28) * 0.5;
        
        let v7  = (v23 + v24) * 0.5;
        let v11 = (v21 + v22) * 0.5;
        let v13 = (v23 - v24) * 0.5;
        let v17 = (v21 - v22) * 0.5;
        
        let v8 = (v15 + v16) * 0.5;
        let v9 = (v15 - v16) * 0.5;

        const F: f32 = 1.0 / (A_2 * A_5 - A_2 * A_4 - A_4 * A_5);
        
        let v18 = (v19 - v20) * A_5;  // Different from original
        let v12 = (v19 * A_4 - v18) * F;
        let v14 = (v18 - v20 * A_2) * F;
        
        let v6 = v14 - v7;
        let v5 = v13 * RA_3 - v6;
        let v4 = -v5 - v12;
        let v10 = v17 * RA_1 - v11;
        
        let v0 = (v8 + v11) * 0.5;
        let v1 = (v9 + v10) * 0.5;
        let v2 = (v9 - v10) * 0.5;
        let v3 = (v8 - v11) * 0.5;
        
        vector[0] = (v0 + v7) * 0.5;
        vector[1] = (v1 + v6) * 0.5;
        vector[2] = (v2 + v5) * 0.5;
        vector[3] = (v3 + v4) * 0.5;
        vector[4] = (v3 - v4) * 0.5;
        vector[5] = (v2 - v5) * 0.5;
        vector[6] = (v1 - v6) * 0.5;
        vector[7] = (v0 - v7) * 0.5;
    }
}