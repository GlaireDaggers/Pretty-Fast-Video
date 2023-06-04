pub const FP_BITS: i32 = 8;

/// Scale factors to be applied to coefficients at encode & decode time, in 24.8 fixed point
pub static DCT_SCALE_FACTOR: [i32;64] = [
    32, 37, 34, 26, 32, 26, 34, 37,
    37, 43, 39, 31, 37, 31, 39, 43,
    34, 39, 35, 28, 34, 28, 35, 39,
    26, 31, 28, 22, 26, 22, 28, 31,
    32, 37, 34, 26, 32, 26, 34, 37,
    26, 31, 28, 22, 26, 22, 28, 31,
    34, 39, 35, 28, 34, 28, 35, 39,
    37, 43, 39, 31, 37, 31, 39, 43,
];

/// Quantization table for intra-frames (I-Frames)
pub static Q_TABLE_INTRA: [i32;64] = [
    8, 16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83,
];

/// Quantization table for inter-frames (P-Frames)
pub static Q_TABLE_INTER: [i32;64] = [
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
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
    pub m: [i32;64]
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
        DctMatrix8x8 { m: [0;64] }
    }

    pub fn decode(src: &DctQuantizedMatrix8x8, q_table: &[i32;64]) -> DctMatrix8x8 {
        let mut result = DctMatrix8x8 { m: [0;64] };

        for idx in INV_ZIGZAG_TABLE {
            let n = src.m[idx] as i32 * DCT_SCALE_FACTOR[idx];
            let d = q_table[idx];

            result.m[idx] = n * d;
        }

        result
    }

    pub fn encode(self: &mut DctMatrix8x8, q_table: &[i32;64]) -> DctQuantizedMatrix8x8 {
        let mut result = DctQuantizedMatrix8x8 { m: [0;64] };

        for idx in ZIGZAG_TABLE {
            let n = (self.m[idx] * DCT_SCALE_FACTOR[idx]) >> (FP_BITS * 2);
            let d = q_table[idx];

            result.m[idx] = (n / d) as i16;
        }

        result
    }

    pub fn get_row(self: &DctMatrix8x8, index: usize) -> [i32;8] {
        assert!(index < 8);
        let row_offset = index * 8;

        let mut result = [0;8];
        result.copy_from_slice(&self.m[row_offset..row_offset+8]);

        result
    }

    pub fn set_row(self: &mut DctMatrix8x8, index: usize, row: [i32;8]) {
        assert!(index < 8);
        let row_offset = index * 8;

        self.m[row_offset..row_offset+8].copy_from_slice(&row);
    }
    
    pub fn get_column(self: &DctMatrix8x8, index: usize) -> [i32;8] {
        assert!(index < 8);

        let mut result = [0;8];

        for row in 0..8 {
            result[row] = self.m[index + (row * 8)];
        }

        result
    }

    pub fn set_column(self: &mut DctMatrix8x8, index: usize, column: [i32;8]) {
        assert!(index < 8);

        for row in 0..8 {
            self.m[index + (row * 8)] = column[row];
        }
    }

    /// Perform an in-place DCT transformation of each row of this matrix
    pub fn dct_transform_rows(self: &mut DctMatrix8x8) {
        for idx in 0..8 {
            let mut row = self.get_row(idx);
            DctMatrix8x8::fdct(&mut row);
            self.set_row(idx, row);
        }
    }

    /// Perform an in-place DCT transformation of each column of this matrix
    pub fn dct_transform_columns(self: &mut DctMatrix8x8) {
        for idx in 0..8 {
            let mut column = self.get_column(idx);
            DctMatrix8x8::fdct(&mut column);
            self.set_column(idx, column);
        }
    }

    /// Perform an in-place inverse DCT transformation of each row of this matrix
    pub fn dct_inverse_transform_rows(self: &mut DctMatrix8x8) {
        for idx in 0..8 {
            let mut row = self.get_row(idx);
            DctMatrix8x8::idct(&mut row);
            self.set_row(idx, row);
        }
    }

    /// Perform an in-place inverse DCT transformation of each column of this matrix
    pub fn dct_inverse_transform_columns(self: &mut DctMatrix8x8) {
        for idx in 0..8 {
            let mut column = self.get_column(idx);
            DctMatrix8x8::idct(&mut column);
            self.set_column(idx, column);
        }
    }

    // adapted from https://fgiesen.wordpress.com/2013/11/04/bink-2-2-integer-dct-design-part-1/

    pub fn fdct(vector: &mut [i32;8]) {
        // extract rows
        let i0 = vector[0];
        let i1 = vector[1];
        let i2 = vector[2];
        let i3 = vector[3];
        let i4 = vector[4];
        let i5 = vector[5];
        let i6 = vector[6];
        let i7 = vector[7];

        // stage 1 - 8A
        let a0 = i0 + i7;
        let a1 = i1 + i6;
        let a2 = i2 + i5;
        let a3 = i3 + i4;
        let a4 = i0 - i7;
        let a5 = i1 - i6;
        let a6 = i2 - i5;
        let a7 = i3 - i4;

        // even stage 2 - 4A
        let b0 = a0 + a3;
        let b1 = a1 + a2;
        let b2 = a0 - a3;
        let b3 = a1 - a2;

        // even stage 3 - 6A 4S
        let c0 = b0 + b1;
        let c1 = b0 - b1;
        let c2 = b2 + b2/4 + b3/2;
        let c3 = b2/2 - b3 - b3/4;

        // odd stage 2 - 12A 8S
        // NB a4/4 and a7/4 are each used twice, so this really is 8 shifts, not 10.
        let b4 = a7/4 + a4 + a4/4 - a4/16;
        let b7 = a4/4 - a7 - a7/4 + a7/16;
        let b5 = a5 + a6 - a6/4 - a6/16;
        let b6 = a6 - a5 + a5/4 + a5/16;

        // odd stage 3 - 4A
        let c4 = b4 + b5;
        let c5 = b4 - b5;
        let c6 = b6 + b7;
        let c7 = b6 - b7;

        // odd stage 4 - 2A
        let d4 = c4;
        let d5 = c5 + c7;
        let d6 = c5 - c7;
        let d7 = c6;

        // permute/output
        vector[0] = c0;
        vector[1] = d4;
        vector[2] = c2;
        vector[3] = d6;
        vector[4] = c1;
        vector[5] = d5;
        vector[6] = c3;
        vector[7] = d7;

        // total: 36A 12S
    }

    pub fn idct(vector: &mut [i32;8]) {
        // extract rows (with input permutation)
        let c0 = vector[0];
        let d4 = vector[1];
        let c2 = vector[2];
        let d6 = vector[3];
        let c1 = vector[4];
        let d5 = vector[5];
        let c3 = vector[6];
        let d7 = vector[7];

        // odd stage 4
        let c4 = d4;
        let c5 = d5 + d6;
        let c7 = d5 - d6;
        let c6 = d7;

        // odd stage 3
        let b4 = c4 + c5;
        let b5 = c4 - c5;
        let b6 = c6 + c7;
        let b7 = c6 - c7;

        // even stage 3
        let b0 = c0 + c1;
        let b1 = c0 - c1;
        let b2 = c2 + c2/4 + c3/2;
        let b3 = c2/2 - c3 - c3/4;

        // odd stage 2
        let a4 = b7/4 + b4 + b4/4 - b4/16;
        let a7 = b4/4 - b7 - b7/4 + b7/16;
        let a5 = b5 - b6 + b6/4 + b6/16;
        let a6 = b6 + b5 - b5/4 - b5/16;

        // even stage 2
        let a0 = b0 + b2;
        let a1 = b1 + b3;
        let a2 = b1 - b3;
        let a3 = b0 - b2;

        // stage 1
        vector[0] = a0 + a4;
        vector[1] = a1 + a5;
        vector[2] = a2 + a6;
        vector[3] = a3 + a7;
        vector[4] = a3 - a7;
        vector[5] = a2 - a6;
        vector[6] = a1 - a5;
        vector[7] = a0 - a4;

        // total: 36A 12S
    }
}