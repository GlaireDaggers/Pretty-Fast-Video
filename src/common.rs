pub const PFV_MAGIC: &[u8] = b"PFVIDEO\0";
pub const PFV_VERSION: u32 = 100;

use crate::{dct::{DctQuantizedMatrix8x8, DctMatrix8x8}, plane::VideoPlane};

#[cfg(feature = "multithreading")]
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct EncodedMacroBlock {
    pub subblocks: [DctQuantizedMatrix8x8;4]
}

#[derive(Clone, Copy)]
pub struct DeltaEncodedMacroBlock {
    pub motion_x: i8,
    pub motion_y: i8,
    pub subblocks: Option<[DctQuantizedMatrix8x8;4]>
}

pub struct MacroBlock {
    pub pixels: [u8;256]
}

pub enum EncodedFrame {
    IFrame (EncodedIFrame),
    PFrame (EncodedPFrame),
    DropFrame,
}

pub struct EncodedIFrame {
    pub y: EncodedIPlane,
    pub u: EncodedIPlane,
    pub v: EncodedIPlane,
}

pub struct EncodedPFrame {
    pub y: EncodedPPlane,
    pub u: EncodedPPlane,
    pub v: EncodedPPlane,
}

pub struct EncodedIPlane {
    pub width: usize,
    pub height: usize,
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub blocks: Vec<EncodedMacroBlock>,
}

pub struct EncodedPPlane {
    pub width: usize,
    pub height: usize,
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub blocks: Vec<DeltaEncodedMacroBlock>,
}

impl MacroBlock {
    pub fn new() -> MacroBlock {
        MacroBlock { pixels: [0;256] }
    }

    pub fn blit_subblock(self: &mut MacroBlock, src: &[u8;64], dx: usize, dy: usize) {
        for row in 0..8 {
            let dest_row = row + dy;
            let src_offset = row * 8;
            let dst_offset = (dest_row * 16) + dx;

            self.pixels[dst_offset..(dst_offset + 8)].copy_from_slice(&src[src_offset..(src_offset + 8)]);
        }
    }

    pub fn apply_residuals(self: &mut MacroBlock, from: &MacroBlock) {
        for (delta, pixel) in self.pixels.iter_mut().zip(from.pixels) {
            let d = (*delta as i16 - 128) * 2;
            let p = pixel as i16;
            *delta = (p + d).clamp(0, 255) as u8;
        }
    }
}

impl VideoPlane {
    fn calc_residuals(from: &VideoPlane, to: &VideoPlane) -> VideoPlane {
        debug_assert!(from.width == to.width && from.height == to.height);

        let mut residuals = VideoPlane::new(from.width, from.height);
        residuals.pixels.copy_from_slice(&from.pixels);

        for (f, t) in residuals.pixels.iter_mut().zip(&to.pixels) {
            let delta = *f as i16 - *t as i16;
            *f = ((delta / 2) + 128).clamp(0, 255) as u8;
        }

        residuals
    }

    fn calc_error(from: &VideoPlane, to: &VideoPlane, ref_lms: f32) -> f32 {
        assert!(from.width == to.width && from.height == to.height);

        let mut sum = 0.0;

        for (_, (a, b)) in from.pixels.iter().zip(&to.pixels).enumerate() {
            let diff = *a as f32 - *b as f32;
            sum += diff * diff;
            if sum >= ref_lms {
                return sum;
            }
        }

        return sum;
    }

    fn encode_block(src: &VideoPlane, q_table: &[f32;64]) -> EncodedMacroBlock {
        debug_assert!(src.width == 16 && src.height == 16);

        // split into 4 subblocks and encode each one
        let subblocks = [
            VideoPlane::encode_subblock(&src.get_slice(0, 0, 8, 8), q_table),
            VideoPlane::encode_subblock(&src.get_slice(8, 0, 8, 8), q_table),
            VideoPlane::encode_subblock(&src.get_slice(0, 8, 8, 8), q_table),
            VideoPlane::encode_subblock(&src.get_slice(8, 8, 8, 8), q_table)];

        EncodedMacroBlock { subblocks: subblocks }
    }

    fn encode_block_delta(src: &VideoPlane, refplane: &VideoPlane, bx: usize, by: usize, q_table: &[f32;64], px_err: f32) -> DeltaEncodedMacroBlock {
        debug_assert!(src.width == 16 && src.height == 16);

        // brute force search around block pos to find delta which minimizes error
        let mut best_dx = 0;
        let mut best_dy = 0;
        let mut best_err = f32::INFINITY;

        let min_err = px_err * px_err * 256.0;

        for my in -16..16 {
            let offsy = by as i32 + my;
            if offsy < 0 || offsy > refplane.height as i32 - 16 {
                continue;
            }
            for mx in -16..16 {
                let offsx = bx as i32 + mx;
                if offsx < 0 || offsx > refplane.width as i32 - 16 {
                    continue;
                }

                let slice = refplane.get_slice(offsx as usize, offsy as usize, 16, 16);
                let err = VideoPlane::calc_error(src, &slice, best_err);

                if err < best_err {
                    best_err = err;
                    best_dx = mx;
                    best_dy = my;
                }
            }
        }

        let prev_block = refplane.get_slice((bx as i32 + best_dx) as usize, (by as i32 + best_dy) as usize, 16, 16);

        // if the best delta is small enough, skip coefficients
        if best_err <= min_err {
            DeltaEncodedMacroBlock { motion_x: best_dx as i8, motion_y: best_dy as i8, subblocks: None }
        } else {
            // generate delta values
            let delta_block = VideoPlane::calc_residuals(src, &prev_block);

            // split into 4 subblocks and encode each one
            let subblocks = [
                VideoPlane::encode_subblock(&delta_block.get_slice(0, 0, 8, 8), q_table),
                VideoPlane::encode_subblock(&delta_block.get_slice(8, 0, 8, 8), q_table),
                VideoPlane::encode_subblock(&delta_block.get_slice(0, 8, 8, 8), q_table),
                VideoPlane::encode_subblock(&delta_block.get_slice(8, 8, 8, 8), q_table)];

            DeltaEncodedMacroBlock { motion_x: best_dx as i8, motion_y: best_dy as i8, subblocks: Some(subblocks) }
        }
    }
    
    fn decode_block(src: &EncodedMacroBlock, q_table: &[f32;64]) -> MacroBlock {
        let subblocks = [
            VideoPlane::decode_subblock(&src.subblocks[0], q_table),
            VideoPlane::decode_subblock(&src.subblocks[1], q_table),
            VideoPlane::decode_subblock(&src.subblocks[2], q_table),
            VideoPlane::decode_subblock(&src.subblocks[3], q_table)];

        let mut block = MacroBlock::new();
        block.blit_subblock(&subblocks[0], 0, 0);
        block.blit_subblock(&subblocks[1], 8, 0);
        block.blit_subblock(&subblocks[2], 0, 8);
        block.blit_subblock(&subblocks[3], 8, 8);

        block
    }

    fn decode_block_delta(src: &DeltaEncodedMacroBlock, refplane: &VideoPlane, bx: usize, by: usize, q_table: &[f32;64]) -> MacroBlock {
        let sx = bx as i32 + src.motion_x as i32;
        let sy = by as i32 + src.motion_y as i32;

        debug_assert!(sx >= 0 && sx <= refplane.width as i32 - 16);
        debug_assert!(sy >= 0 && sy <= refplane.height as i32 - 16);

        let prev_block = refplane.get_block(sx as usize, sy as usize);

        match src.subblocks {
            Some(subblocks) => {
                let subblocks = [
                    VideoPlane::decode_subblock(&subblocks[0], q_table),
                    VideoPlane::decode_subblock(&subblocks[1], q_table),
                    VideoPlane::decode_subblock(&subblocks[2], q_table),
                    VideoPlane::decode_subblock(&subblocks[3], q_table)];

                let mut block = MacroBlock::new();
                block.blit_subblock(&subblocks[0], 0, 0);
                block.blit_subblock(&subblocks[1], 8, 0);
                block.blit_subblock(&subblocks[2], 0, 8);
                block.blit_subblock(&subblocks[3], 8, 8);

                block.apply_residuals(&prev_block);

                return block;
            }
            None => {
                return prev_block;
            }
        };
    }

    fn encode_subblock(src: &VideoPlane, q_table: &[f32;64]) -> DctQuantizedMatrix8x8 {
        assert!(src.width == 8 && src.height == 8);

        let mut dct = DctMatrix8x8::new();
        let cell_px: Vec<f32> = src.pixels.iter().map(|x| (*x as f32) - 128.0).collect();
        dct.m.copy_from_slice(&cell_px);

        dct.dct_transform_rows();
        dct.dct_transform_columns();

        dct.encode(q_table)
    }

    fn decode_subblock(src: &DctQuantizedMatrix8x8, q_table: &[f32;64]) -> [u8;64] {
        let mut dct = DctMatrix8x8::decode(src, q_table);
        dct.dct_inverse_transform_columns();
        dct.dct_inverse_transform_rows();

        let mut result = [0;64];
        
        for (idx, px) in dct.m.iter().enumerate() {
            result[idx] = (*px + 128.0) as u8;
        }

        result
    }

    pub fn get_block(self: &VideoPlane, sx: usize, sy: usize) -> MacroBlock {
        let mut dest: MacroBlock = MacroBlock { pixels: [0;256] };

        for row in 0..16 {
            let src_row = row + sy;
            let src_offset = (src_row * self.width) + sx;
            let dst_offset = row * 16;

            dest.pixels[dst_offset..(dst_offset + 16)].copy_from_slice(&self.pixels[src_offset..(src_offset + 16)]);
        }

        dest
    }

    pub fn blit_block(self: &mut VideoPlane, block: &MacroBlock, dx: usize, dy: usize) {
        for row in 0..16 {
            let dest_row = row + dy;
            let src_offset = row * 16;
            let dst_offset = (dest_row * self.width) + dx;

            self.pixels[dst_offset..(dst_offset + 16)].copy_from_slice(&block.pixels[src_offset..(src_offset + 16)]);
        }
    }

    #[cfg(feature = "multithreading")]
    pub fn encode_plane(self: &VideoPlane, q_table: &[f32;64], clear_color: u8) -> EncodedIPlane {
        let pad_width: usize = self.width + (16 - (self.width % 16)) % 16;
        let pad_height = self.height + (16 - (self.height % 16)) % 16;
        let mut img_copy = VideoPlane::new(pad_width, pad_height);
        img_copy.pixels.fill(clear_color);
        img_copy.blit(self, 0, 0, 0, 0, self.width, self.height);

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let mut blocks: Vec<VideoPlane> = Vec::with_capacity(blocks_wide * blocks_high);

        // split image plane into 16x16 macroblocks
        for block_y in 0..blocks_high {
            for block_x in 0..blocks_wide {
                let mut block = VideoPlane::new(16, 16);
                block.blit(&img_copy, 0, 0, block_x * 16, block_y * 16, 16, 16);
                blocks.push(block);
            }
        }

        // encode each macroblock in parallel
        let enc_result: Vec<_> = blocks.par_iter().map(|x| {
            VideoPlane::encode_block(x, q_table)
        }).collect();

        EncodedIPlane { width: pad_width, height: pad_height, blocks_wide: blocks_wide, blocks_high: blocks_high, blocks: enc_result }
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn encode_plane(self: &VideoPlane, q_table: &[f32;64], clear_color: u8) -> EncodedIPlane {
        let pad_width: usize = self.width + (16 - (self.width % 16)) % 16;
        let pad_height = self.height + (16 - (self.height % 16)) % 16;
        let mut img_copy = VideoPlane::new(pad_width, pad_height);
        img_copy.pixels.fill(clear_color);
        img_copy.blit(self, 0, 0, 0, 0, self.width, self.height);

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let mut blocks: Vec<VideoPlane> = Vec::with_capacity(blocks_wide * blocks_high);

        // split image plane into 16x16 macroblocks
        for block_y in 0..blocks_high {
            for block_x in 0..blocks_wide {
                let mut block = VideoPlane::new(16, 16);
                block.blit(&img_copy, 0, 0, block_x * 16, block_y * 16, 16, 16);
                blocks.push(block);
            }
        }

        // encode each macroblock in parallel
        let enc_result: Vec<_> = blocks.iter().map(|x| {
            VideoPlane::encode_block(x, q_table)
        }).collect();

        EncodedIPlane { width: pad_width, height: pad_height, blocks_wide: blocks_wide, blocks_high: blocks_high, blocks: enc_result }
    }

    #[cfg(feature = "multithreading")]
    pub fn encode_plane_delta(self: &VideoPlane, refplane: &VideoPlane, q_table: &[f32;64], px_err: f32, clear_color: u8) -> EncodedPPlane {
        let pad_width: usize = self.width + (16 - (self.width % 16)) % 16;
        let pad_height = self.height + (16 - (self.height % 16)) % 16;
        let mut img_copy = VideoPlane::new(pad_width, pad_height);
        img_copy.pixels.fill(clear_color);
        img_copy.blit(self, 0, 0, 0, 0, self.width, self.height);

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let mut blocks: Vec<_> = Vec::with_capacity(blocks_wide * blocks_high);

        // split image plane into 16x16 macroblocks
        for block_y in 0..blocks_high {
            for block_x in 0..blocks_wide {
                let mut block = VideoPlane::new(16, 16);
                block.blit(&img_copy, 0, 0, block_x * 16, block_y * 16, 16, 16);
                blocks.push((block, block_x * 16, block_y * 16));
            }
        }

        // encode each macroblock in parallel
        let enc_result: Vec<_> = blocks.par_iter().map(|(block, bx, by)| {
            VideoPlane::encode_block_delta(block, refplane, *bx, *by, q_table, px_err)
        }).collect();

        EncodedPPlane { width: pad_width, height: pad_height, blocks_wide: blocks_wide, blocks_high: blocks_high, blocks: enc_result }
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn encode_plane_delta(self: &VideoPlane, refplane: &VideoPlane, q_table: &[f32;64], clear_color: u8) -> EncodedPPlane {
        let pad_width: usize = self.width + (16 - (self.width % 16)) % 16;
        let pad_height = self.height + (16 - (self.height % 16)) % 16;
        let mut img_copy = VideoPlane::new(pad_width, pad_height);
        img_copy.pixels.fill(clear_color);
        img_copy.blit(self, 0, 0, 0, 0, self.width, self.height);

        let blocks_wide = pad_width / 16;
        let blocks_high = pad_height / 16;

        let mut blocks: Vec<_> = Vec::with_capacity(blocks_wide * blocks_high);

        // split image plane into 16x16 macroblocks
        for block_y in 0..blocks_high {
            for block_x in 0..blocks_wide {
                let mut block = VideoPlane::new(16, 16);
                block.blit(&img_copy, 0, 0, block_x * 16, block_y * 16, 16, 16);
                blocks.push((block, block_x * 16, block_y * 16));
            }
        }

        // encode each macroblock in parallel
        let enc_result: Vec<_> = blocks.iter().map(|(block, bx, by)| {
            VideoPlane::encode_block_delta(block, refplane, *bx, *by, q_table)
        }).collect();

        EncodedPPlane { width: pad_width, height: pad_height, blocks_wide: blocks_wide, blocks_high: blocks_high, blocks: enc_result }
    }

    #[cfg(feature = "multithreading")]
    pub fn decode_plane(src: &EncodedIPlane, q_table: &[f32;64]) -> VideoPlane {
        let mut plane = VideoPlane::new(src.blocks_wide * 16, src.blocks_high * 16);

        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_par_iter().map(|x| {
            VideoPlane::decode_block(&src.blocks[x], q_table)
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                plane.blit_block(block, block_x * 16, block_y * 16);
            }
        }

        plane
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn decode_plane(src: &EncodedIPlane, q_table: &[f32;64]) -> VideoPlane {
        let mut plane = VideoPlane::new(src.blocks_wide * 16, src.blocks_high * 16);

        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_iter().map(|x| {
            VideoPlane::decode_block(&src.blocks[x], q_table)
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                plane.blit_block(block, block_x * 16, block_y * 16);
            }
        }

        plane
    }

    #[cfg(feature = "multithreading")]
    pub fn decode_plane_delta(src: &EncodedPPlane, refplane: &VideoPlane, q_table: &[f32;64]) -> VideoPlane {
        let mut plane = VideoPlane::new(src.blocks_wide * 16, src.blocks_high * 16);

        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_par_iter().map(|x| {
            let bx = x % src.blocks_wide;
            let by = x / src.blocks_wide;
            VideoPlane::decode_block_delta(&src.blocks[x], refplane, bx * 16, by * 16, q_table)
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                plane.blit_block(block, block_x * 16, block_y * 16);
            }
        }

        plane
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn decode_plane_delta(src: &EncodedPPlane, refplane: &VideoPlane, q_table: &[f32;64]) -> VideoPlane {
        let mut plane = VideoPlane::new(src.blocks_wide * 16, src.blocks_high * 16);

        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_iter().map(|x| {
            let bx = x % src.blocks_wide;
            let by = x / src.blocks_wide;
            VideoPlane::decode_block_delta(&src.blocks[x], refplane, bx * 16, by * 16, q_table)
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                plane.blit_block(block, block_x * 16, block_y * 16);
            }
        }

        plane
    }

    #[cfg(feature = "multithreading")]
    pub fn decode_plane_into(src: &EncodedIPlane, q_table: &[f32;64], target: &mut VideoPlane) {
        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_par_iter().map(|x| {
            VideoPlane::decode_block(&src.blocks[x], q_table)
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                target.blit_block(block, block_x * 16, block_y * 16);
            }
        }
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn decode_plane_into(src: &EncodedIPlane, q_table: &[f32;64], target: &mut VideoPlane) {
        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_iter().map(|x| {
            VideoPlane::decode_block(&src.blocks[x], q_table)
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                target.blit_block(block, block_x * 16, block_y * 16);
            }
        }
    }

    #[cfg(feature = "multithreading")]
    pub fn decode_plane_delta_into(src: &EncodedPPlane, refplane: &mut VideoPlane, q_table: &[f32;64]) {
        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_par_iter().map(|x| {
            let bx = x % src.blocks_wide;
            let by = x / src.blocks_wide;
            VideoPlane::decode_block_delta(&src.blocks[x], refplane, bx * 16, by * 16, q_table)
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                refplane.blit_block(block, block_x * 16, block_y * 16);
            }
        }
    }

    #[cfg(not(feature = "multithreading"))]
    pub fn decode_plane_delta_into(src: &EncodedPPlane, refplane: &mut VideoPlane, q_table: &[f32;64]) {
        let total_blocks = src.blocks_wide * src.blocks_high;
        let results: Vec<_> = (0..total_blocks).into_iter().map(|x| {
            let bx = x % src.blocks_wide;
            let by = x / src.blocks_wide;
            VideoPlane::decode_block_delta(&src.blocks[x], refplane, bx * 16, by * 16, q_table)
        }).collect();

        for block_y in 0..src.blocks_high {
            for block_x in 0..src.blocks_wide {
                let block = &results[block_x + (block_y * src.blocks_wide)];
                refplane.blit_block(block, block_x * 16, block_y * 16);
            }
        }
    }

    pub fn reduce(self: &VideoPlane) -> VideoPlane {
        let mut new_slice = VideoPlane::new(self.width / 2, self.height / 2);

        for iy in 0..new_slice.height {
            for ix in 0..new_slice.width {
                let sx = ix * 2;
                let sy = iy * 2;

                new_slice.pixels[ix + (iy * new_slice.width)] = self.pixels[sx + (sy * self.width)];
            }
        }

        new_slice
    }

    pub fn double(self: &VideoPlane) -> VideoPlane {
        let mut new_slice = VideoPlane::new(self.width * 2, self.height * 2);

        for iy in 0..self.height {
            for ix in 0..self.width {
                let dx = ix * 2;
                let dy = iy * 2;
                let d_idx = dx + (dy * new_slice.width);
                let px = self.pixels[ix + (iy * self.width)];

                new_slice.pixels[d_idx] = px;
                new_slice.pixels[d_idx + 1] = px;
                new_slice.pixels[d_idx + new_slice.width] = px;
                new_slice.pixels[d_idx + new_slice.width + 1] = px;
            }
        }

        new_slice
    }
}