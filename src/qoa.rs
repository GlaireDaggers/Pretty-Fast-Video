// adapted from https://github.com/phoboslab/qoa/blob/master/qoa.h

//pub const QOA_MIN_FILESIZE: usize = 16;
//pub const QOA_MAX_CHANNELS: usize = 8;

pub const QOA_SLICE_LEN: usize = 20;
pub const QOA_SLICES_PER_FRAME: usize = 256;
pub const QOA_FRAME_LEN: usize = QOA_SLICES_PER_FRAME * QOA_SLICE_LEN;
pub const QOA_LMS_LEN: usize = 4;
//pub const QOA_MAGIC: u32 = 0x716f6166;

pub static QOA_QUANT_TABLE: [i32;17] = [
    7, 7, 7, 5, 5, 3, 3, 1, /* -8..-1 */
	0,                      /*  0     */
	0, 2, 2, 4, 4, 6, 6, 6  /*  1.. 8 */
];

//pub static QOA_SCALEFACTOR_TABLE: [i32;16] = [
//    1, 7, 21, 45, 84, 138, 211, 304, 421, 562, 731, 928, 1157, 1419, 1715, 2048
//];

pub static QOA_RECIPROCAL_TABLE: [i32;16] = [
	65536, 9363, 3121, 1457, 781, 475, 311, 216, 156, 117, 90, 71, 57, 47, 39, 32
];

pub static QOA_DEQUANT_TABLE: [[i32;8];16] = [
    [   1,    -1,    3,    -3,    5,    -5,     7,     -7],
	[   5,    -5,   18,   -18,   32,   -32,    49,    -49],
	[  16,   -16,   53,   -53,   95,   -95,   147,   -147],
	[  34,   -34,  113,  -113,  203,  -203,   315,   -315],
	[  63,   -63,  210,  -210,  378,  -378,   588,   -588],
	[ 104,  -104,  345,  -345,  621,  -621,   966,   -966],
	[ 158,  -158,  528,  -528,  950,  -950,  1477,  -1477],
	[ 228,  -228,  760,  -760, 1368, -1368,  2128,  -2128],
	[ 316,  -316, 1053, -1053, 1895, -1895,  2947,  -2947],
	[ 422,  -422, 1405, -1405, 2529, -2529,  3934,  -3934],
	[ 548,  -548, 1828, -1828, 3290, -3290,  5117,  -5117],
	[ 696,  -696, 2320, -2320, 4176, -4176,  6496,  -6496],
	[ 868,  -868, 2893, -2893, 5207, -5207,  8099,  -8099],
	[1064, -1064, 3548, -3548, 6386, -6386,  9933,  -9933],
	[1286, -1286, 4288, -4288, 7718, -7718, 12005, -12005],
	[1536, -1536, 5120, -5120, 9216, -9216, 14336, -14336],
];

#[derive(Clone, Copy)]
pub struct LMS {
    pub history: [i32;QOA_LMS_LEN],
    pub weight: [i32;QOA_LMS_LEN],
}

pub struct EncodedAudioFrame {
    pub samples: usize,
    pub lmses: Vec<LMS>,
    pub slices: Vec<u64>,
}

impl LMS {
    pub fn update(self: &mut LMS, sample: i32, residual: i32) {
        let delta = residual >> 4;
        for i in 0..QOA_LMS_LEN {
            self.weight[i] += if self.history[i] < 0 { -delta } else { delta };
        }
        for i in 0..QOA_LMS_LEN-1 {
            self.history[i] = self.history[i + 1];
        }
        self.history[QOA_LMS_LEN - 1] = sample;
    }
}

pub fn qoa_div(v: i32, scalefactor: usize) -> i32 {
    let reciprocal = QOA_RECIPROCAL_TABLE[scalefactor];
	let n = (v.wrapping_mul(reciprocal) + (1 << 15)) >> 16;
	let n = n + (((v > 0) as i32) - ((v < 0) as i32)) - (((n > 0) as i32) - ((n < 0) as i32)); /* round away from 0 */
	return n;
}

pub fn qoa_lms_predict(lms: LMS) -> i32 {
    let mut prediction = 0;
	for i in 0..QOA_LMS_LEN {
		prediction += lms.weight[i] * lms.history[i];
	}
	return prediction >> 13;
}

//pub fn calc_frame_size(channels: usize, slices: usize) -> usize {
//    8 + QOA_LMS_LEN * 4 * channels + 8 * slices * channels
//}