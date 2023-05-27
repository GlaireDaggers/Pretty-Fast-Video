use ocl::ProQue;

pub fn build_decoder_queue(width: usize, height: usize) -> ProQue {
    let src = r#"
// Q.enqueueNDRangeKernel(K, NullRange, NDRange(1920 / 16, 1080 / 16), NDRange(1, 1))

const float S_0 = 0.353553390593273762200422;
const float S_1 = 0.254897789552079584470970;
const float S_2 = 0.270598050073098492199862;
const float S_3 = 0.300672443467522640271861;
const float S_4 = 0.353553390593273762200422;
const float S_5 = 0.449988111568207852319255;
const float S_6 = 0.653281482438188263928322;
const float S_7 = 1.281457723870753089398043;

const float RS_0 = 1.0 / S_0;
const float RS_1 = 1.0 / S_1;
const float RS_2 = 1.0 / S_2;
const float RS_3 = 1.0 / S_3;
const float RS_4 = 1.0 / S_4;
const float RS_5 = 1.0 / S_5;
const float RS_6 = 1.0 / S_6;
const float RS_7 = 1.0 / S_7;

const float A_1 = 0.707106781186547524400844;
const float A_2 = 0.541196100146196984399723;
const float A_3 = 0.707106781186547524400844;
const float A_4 = 1.306562964876376527856643;
const float A_5 = 0.382683432365089771728460;

const float RA_1 = 1.0 / A_1;
const float RA_2 = 1.0 / A_2;
const float RA_3 = 1.0 / A_3;
const float RA_4 = 1.0 / A_4;
const float RA_5 = 1.0 / A_5;

__constant int INV_ZIGZAG_TABLE[64] = {
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 
33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
};

void idct(float* vec, int stride) {
	float v15 = vector[0 * stride] * RS_0;
	float v26 = vector[1 * stride] * RS_1;
	float v21 = vector[2 * stride] * RS_2;
	float v28 = vector[3 * stride] * RS_3;
	float v16 = vector[4 * stride] * RS_4;
	float v25 = vector[5 * stride] * RS_5;
	float v22 = vector[6 * stride] * RS_6;
	float v27 = vector[7 * stride] * RS_7;
	
	float v19 = (v25 - v28) * 0.5;
	float v20 = (v26 - v27) * 0.5;
	float v23 = (v26 + v27) * 0.5;
	float v24 = (v25 + v28) * 0.5;
	
	float v7  = (v23 + v24) * 0.5;
	float v11 = (v21 + v22) * 0.5;
	float v13 = (v23 - v24) * 0.5;
	float v17 = (v21 - v22) * 0.5;
	
	float v8 = (v15 + v16) * 0.5;
	float v9 = (v15 - v16) * 0.5;

	const float F = 1.0 / (A_2 * A_5 - A_2 * A_4 - A_4 * A_5);
	
	float v18 = (v19 - v20) * A_5;  // Different from original
	float v12 = (v19 * A_4 - v18) * F;
	float v14 = (v18 - v20 * A_2) * F;
	
	float v6 = v14 - v7;
	float v5 = v13 * RA_3 - v6;
	float v4 = -v5 - v12;
	float v10 = v17 * RA_1 - v11;
	
	float v0 = (v8 + v11) * 0.5;
	float v1 = (v9 + v10) * 0.5;
	float v2 = (v9 - v10) * 0.5;
	float v3 = (v8 - v11) * 0.5;
	
	vector[0 * stride] = (v0 + v7) * 0.5;
	vector[1 * stride] = (v1 + v6) * 0.5;
	vector[2 * stride] = (v2 + v5) * 0.5;
	vector[3 * stride] = (v3 + v4) * 0.5;
	vector[4 * stride] = (v3 - v4) * 0.5;
	vector[5 * stride] = (v2 - v5) * 0.5;
	vector[6 * stride] = (v1 - v6) * 0.5;
	vector[7 * stride] = (v0 - v7) * 0.5;
}

void dct8x8_decode(float* dct) {
	// inverse dct columns
	for (int i = 0; i < 8; i++) {
		idct(&dct[i], 8);
	}
	
	// inverse dct rows
	for (int i = 0; i < 8; i++) {
		idct(&dct[i * 8], 1);
	}
}

void blit_subblock(float* subblock, int bx, int by, write_only image2d_t out_image) {
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++) {
			write_imagef(int2(bx + x, by + y), subblock[x + (y * 8)]);
		}
	}
}

__kernel void decode_iframe(__global short* in_buffer, __global float* qtable, write_only image2d_t out_image) {
	int block_x = get_global_id(0);
	int block_y = get_global_id(1);
	int blocks_wide = get_global_size(0);
	int blocks_high = get_global_size(1);
	
	int block_index = block_x + (block_y * blocks_wide);
	int block_coeff_offset = block_index * 256;
	
	float subblock_0[64];
	float subblock_1[64];
	float subblock_2[64];
	float subblock_3[64];
	
	// read each subblock
	for (int i = 0; i < 64; i++)
	{
		int rd_index = INV_ZIGZAG_TABLE[i];
		subblock_0[i] = (float)in_buffer[block_coeff_offset + rd_index] * qtable[i];
		subblock_1[i] = (float)in_buffer[block_coeff_offset + 64 + rd_index] * qtable[i];
		subblock_2[i] = (float)in_buffer[block_coeff_offset + 128 + rd_index] * qtable[i];
		subblock_3[i] = (float)in_buffer[block_coeff_offset + 192 + rd_index] * qtable[i];
	}
	
	// decode each subblock
	dct8x8_decode(subblock_0);
	dct8x8_decode(subblock_1);
	dct8x8_decode(subblock_2);
	dct8x8_decode(subblock_3);
	
	int bx = block_x * 16;
	int by = block_y * 16;
	
	// blit subblocks into target image
	blit_subblock(subblock_0, bx, by, out_image);
	blit_subblock(subblock_1, bx + 8, by, out_image);
	blit_subblock(subblock_2, bx, by + 8, out_image);
	blit_subblock(subblock_3, bx + 8, by + 8, out_image);
}
    "#;

    ProQue::builder()
        .src(src)
        .dims((width / 16, height / 16))
        .build().expect("Failed creating OpenCL queue")
}