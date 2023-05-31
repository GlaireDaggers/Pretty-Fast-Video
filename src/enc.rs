use std::io::{Write, Cursor};

use bitstream_io::{BitWriter, BitWrite};
use byteorder::{WriteBytesExt, LittleEndian};

use crate::common::{EncodedIFrame, PFV_MAGIC, PFV_VERSION, EncodedPFrame};
use crate::frame::VideoFrame;
use crate::dct::{Q_TABLE_INTER, Q_TABLE_INTRA};
use crate::plane::VideoPlane;
use crate::rle::{rle_encode, rle_create_huffman, update_table};

pub struct Encoder<W: Write> {
    width: usize,
    height: usize,
    framerate: u32,
    prev_frame: VideoFrame,
    px_err: f32,
    qtable_inter_l: [f32;64],
    qtable_inter_c: [f32;64],
    qtable_intra_l: [f32;64],
    qtable_intra_c: [f32;64],
    writer: W,
    finished: bool,
    #[cfg(feature = "multithreading")]
    threadpool: rayon::ThreadPool
}

impl<W: Write> Drop for Encoder<W> {
    fn drop(&mut self) {
        if !self.finished {
            self.finish().unwrap();
        }   
    }
}

impl<W: Write> Encoder<W> {
    pub fn new(writer: W, width: usize, height: usize, framerate: u32, quality: i32, #[cfg(feature = "multithreading")] num_threads: usize) -> Result<Encoder<W>, std::io::Error> {
        assert!(quality >= 0 && quality <= 10);

        let qscale = quality as f32 * 0.25;
        let px_err = quality as f32 * 1.5;

        #[cfg(feature = "multithreading")]
        let mut enc = {
            Encoder { width: width, height: height, framerate: framerate,
                prev_frame: VideoFrame::new_padded(width, height),
                px_err: px_err,
                qtable_inter_l: Q_TABLE_INTER.map(|x| (x * qscale * 0.5).max(1.0)),
                qtable_inter_c: Q_TABLE_INTER.map(|x| (x * qscale).max(1.0)),
                qtable_intra_l: Q_TABLE_INTRA.map(|x| (x * qscale * 0.5).max(1.0)),
                qtable_intra_c: Q_TABLE_INTRA.map(|x| (x * qscale).max(1.0)),
                writer: writer,
                finished: false,
                threadpool: rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap() }
        };

        #[cfg(not(feature = "multithreading"))]
        let mut enc = {
            Encoder { width: width, height: height, framerate: framerate, samplerate: samplerate, channels: channels,
                prev_frame: VideoFrame::new_padded(width, height),
                px_err: px_err,
                qtable_inter: Q_TABLE_INTER.map(|x| (x * qscale).max(1.0)),
                qtable_intra: Q_TABLE_INTRA.map(|x| (x * qscale).max(1.0)),
                writer: writer,
                finished: false, }
        };

        enc.write_header()?;

        Ok(enc)
    }

    pub fn encode_iframe(self: &mut Encoder<W>, frame: &VideoFrame) -> Result<(), std::io::Error> {
        assert!(frame.width == self.width && frame.height == self.height);
        assert!(frame.plane_y.width == frame.width && frame.plane_y.height == frame.height);
        assert!(frame.plane_u.width == frame.width / 2 && frame.plane_u.height == frame.height / 2);
        assert!(frame.plane_v.width == frame.width / 2 && frame.plane_v.height == frame.height / 2);
        assert!(!self.finished);

        #[cfg(feature = "multithreading")]
        {
            let enc_y = frame.plane_y.encode_plane(&self.qtable_intra_l, 0, &self.threadpool);
            let dec_y = VideoPlane::decode_plane(&enc_y, &self.qtable_intra_l, &self.threadpool);

            let enc_u = frame.plane_u.encode_plane(&self.qtable_intra_c, 128, &self.threadpool);
            let dec_u = VideoPlane::decode_plane(&enc_u, &self.qtable_intra_c, &self.threadpool);

            let enc_v = frame.plane_v.encode_plane(&self.qtable_intra_c, 128, &self.threadpool);
            let dec_v = VideoPlane::decode_plane(&enc_v, &self.qtable_intra_c, &self.threadpool);

            let enc_frame = EncodedIFrame { y: enc_y, u: enc_u, v: enc_v };

            self.prev_frame.plane_y.blit(&dec_y, 0, 0, 0, 0, dec_y.width, dec_y.height);
            self.prev_frame.plane_u.blit(&dec_u, 0, 0, 0, 0, dec_u.width, dec_u.height);
            self.prev_frame.plane_v.blit(&dec_v, 0, 0, 0, 0, dec_v.width, dec_v.height);

            Encoder::<W>::write_iframe_packet(&enc_frame, &mut self.writer)?;
        }

        #[cfg(not(feature = "multithreading"))]
        {
            let enc_y = frame.plane_y.encode_plane(&self.qtable_intra, 0);
            let dec_y = VideoPlane::decode_plane(&enc_y, &self.qtable_intra);

            let enc_u = frame.plane_u.encode_plane(&self.qtable_intra, 128);
            let dec_u = VideoPlane::decode_plane(&enc_u, &self.qtable_intra);

            let enc_v = frame.plane_v.encode_plane(&self.qtable_intra, 128);
            let dec_v = VideoPlane::decode_plane(&enc_v, &self.qtable_intra);

            let enc_frame = EncodedIFrame { y: enc_y, u: enc_u, v: enc_v };

            self.prev_frame.plane_y.blit(&dec_y, 0, 0, 0, 0, dec_y.width, dec_y.height);
            self.prev_frame.plane_u.blit(&dec_u, 0, 0, 0, 0, dec_u.width, dec_u.height);
            self.prev_frame.plane_v.blit(&dec_v, 0, 0, 0, 0, dec_v.width, dec_v.height);

            Encoder::<W>::write_iframe_packet(&enc_frame, &mut self.writer)?;
        }

        Ok(())
    }

    pub fn encode_pframe(self: &mut Encoder<W>, frame: &VideoFrame) -> Result<(), std::io::Error> {
        assert!(frame.width == self.width && frame.height == self.height);
        assert!(frame.plane_y.width == frame.width && frame.plane_y.height == frame.height);
        assert!(frame.plane_u.width == frame.width / 2 && frame.plane_u.height == frame.height / 2);
        assert!(frame.plane_v.width == frame.width / 2 && frame.plane_v.height == frame.height / 2);
        assert!(!self.finished);

        #[cfg(feature = "multithreading")]
        {
            let enc_y = frame.plane_y.encode_plane_delta(&self.prev_frame.plane_y, &self.qtable_inter_l, self.px_err, 0, &self.threadpool);
            let dec_y = VideoPlane::decode_plane_delta(&enc_y, &self.prev_frame.plane_y, &self.qtable_inter_l, &self.threadpool);

            let enc_u = frame.plane_u.encode_plane_delta(&self.prev_frame.plane_u, &self.qtable_inter_c, self.px_err, 128, &self.threadpool);
            let dec_u = VideoPlane::decode_plane_delta(&enc_u, &self.prev_frame.plane_u, &self.qtable_inter_c, &self.threadpool);

            let enc_v = frame.plane_v.encode_plane_delta(&self.prev_frame.plane_v, &self.qtable_inter_c, self.px_err, 128, &self.threadpool);
            let dec_v = VideoPlane::decode_plane_delta(&enc_v, &self.prev_frame.plane_v, &self.qtable_inter_c, &self.threadpool);

            let enc_frame = EncodedPFrame { y: enc_y, u: enc_u, v: enc_v };

            self.prev_frame.plane_y.blit(&dec_y, 0, 0, 0, 0, dec_y.width, dec_y.height);
            self.prev_frame.plane_u.blit(&dec_u, 0, 0, 0, 0, dec_u.width, dec_u.height);
            self.prev_frame.plane_v.blit(&dec_v, 0, 0, 0, 0, dec_v.width, dec_v.height);
            
            Encoder::<W>::write_pframe_packet(&enc_frame, &mut self.writer)?;
        }

        #[cfg(not(feature = "multithreading"))]
        {
            let enc_y = frame.plane_y.encode_plane_delta(&self.prev_frame.plane_y, &self.qtable_inter, self.px_err, 0);
            let dec_y = VideoPlane::decode_plane_delta(&enc_y, &self.prev_frame.plane_y, &self.qtable_inter);

            let enc_u = frame.plane_u.encode_plane_delta(&self.prev_frame.plane_u, &self.qtable_inter, self.px_err, 128);
            let dec_u = VideoPlane::decode_plane_delta(&enc_u, &self.prev_frame.plane_u, &self.qtable_inter);

            let enc_v = frame.plane_v.encode_plane_delta(&self.prev_frame.plane_v, &self.qtable_inter, self.px_err, 128);
            let dec_v = VideoPlane::decode_plane_delta(&enc_v, &self.prev_frame.plane_v, &self.qtable_inter);

            let enc_frame = EncodedPFrame { y: enc_y, u: enc_u, v: enc_v };

            self.prev_frame.plane_y.blit(&dec_y, 0, 0, 0, 0, dec_y.width, dec_y.height);
            self.prev_frame.plane_u.blit(&dec_u, 0, 0, 0, 0, dec_u.width, dec_u.height);
            self.prev_frame.plane_v.blit(&dec_v, 0, 0, 0, 0, dec_v.width, dec_v.height);

            Encoder::<W>::write_pframe_packet(&enc_frame, &mut self.writer)?;
        }

        Ok(())
    }

    pub fn encode_dropframe(self: &mut Encoder<W>) -> Result<(), std::io::Error> {
        assert!(!self.finished);

        Encoder::<W>::write_drop_packet(&mut self.writer)?;
        Ok(())
    }

    pub fn finish(self: &mut Encoder<W>) -> Result<(), std::io::Error> {
        assert!(!self.finished);

        self.finished = true;
        Encoder::write_eof(&mut self.writer)?;
        Ok(())
    }

    fn write_header(self: &mut Encoder<W>) -> Result<(), std::io::Error> {
        // write PGV header
        self.writer.write_all(PFV_MAGIC)?;
        self.writer.write_u32::<LittleEndian>(PFV_VERSION)?;

        self.writer.write_u16::<LittleEndian>(self.width as u16)?;
        self.writer.write_u16::<LittleEndian>(self.height as u16)?;
        self.writer.write_u16::<LittleEndian>(self.framerate as u16)?;

        // write q-tables
        self.writer.write_u16::<LittleEndian>(4)?;

        for v in self.qtable_intra_l {
            self.writer.write_u16::<LittleEndian>(v as u16)?;
        }

        for v in self.qtable_intra_c {
            self.writer.write_u16::<LittleEndian>(v as u16)?;
        }

        for v in self.qtable_inter_l {
            self.writer.write_u16::<LittleEndian>(v as u16)?;
        }

        for v in self.qtable_inter_c {
            self.writer.write_u16::<LittleEndian>(v as u16)?;
        }

        Ok(())
    }

    fn write_eof(writer: &mut W) -> Result<(), std::io::Error> {
        // write packet header
        writer.write_u8(0)?; // packet type = EOF
        writer.write_u32::<LittleEndian>(0)?;

        Ok(())
    }

    fn write_drop_packet(writer: &mut W) -> Result<(), std::io::Error> {
        // write packet header
        writer.write_u8(1)?; // packet type = iframe
        writer.write_u32::<LittleEndian>(0)?;

        Ok(())
    }

    fn write_iframe_packet(f: &EncodedIFrame, writer: &mut W) -> Result<(), std::io::Error> {
        // serialize packet data
        let mut packet_data = Cursor::new(Vec::new());
        let mut bitwriter = BitWriter::endian(&mut packet_data, bitstream_io::LittleEndian);

        // gather RLE-encoded block coefficients for each plane
        let mut block_coeff = Vec::new();
        let mut symbol_table = [0;16];

        for b in &f.y.blocks {
            let mut coeff = Vec::new();
            coeff.extend_from_slice(&b.subblocks[0].m);
            coeff.extend_from_slice(&b.subblocks[1].m);
            coeff.extend_from_slice(&b.subblocks[2].m);
            coeff.extend_from_slice(&b.subblocks[3].m);
            let mut rle_sequence = Vec::new();
            rle_encode(&mut rle_sequence, &coeff);
            update_table(&mut symbol_table, &rle_sequence);

            block_coeff.push(rle_sequence);
        }

        for b in &f.u.blocks {
            let mut coeff = Vec::new();
            coeff.extend_from_slice(&b.subblocks[0].m);
            coeff.extend_from_slice(&b.subblocks[1].m);
            coeff.extend_from_slice(&b.subblocks[2].m);
            coeff.extend_from_slice(&b.subblocks[3].m);
            let mut rle_sequence = Vec::new();
            rle_encode(&mut rle_sequence, &coeff);
            update_table(&mut symbol_table, &rle_sequence);

            block_coeff.push(rle_sequence);
        }

        for b in &f.v.blocks {
            let mut coeff = Vec::new();
            coeff.extend_from_slice(&b.subblocks[0].m);
            coeff.extend_from_slice(&b.subblocks[1].m);
            coeff.extend_from_slice(&b.subblocks[2].m);
            coeff.extend_from_slice(&b.subblocks[3].m);
            let mut rle_sequence = Vec::new();
            rle_encode(&mut rle_sequence, &coeff);
            update_table(&mut symbol_table, &rle_sequence);

            block_coeff.push(rle_sequence);
        }

        // create huffman tree for encoding RLE results
        let tree = rle_create_huffman(&symbol_table);
        let tree_table = tree.get_table();

        // write symbol frequency table
        for i in 0..16 {
            bitwriter.write(8, tree_table[i] as u8)?;
        }

        // we currently create four qtables: two for i-frames (0, 1) and two for p-frames (2, 3)
        // note: (one qtable index per plane)
        bitwriter.write(8, 0_u8)?;
        bitwriter.write(8, 1_u8)?;
        bitwriter.write(8, 1_u8)?;

        // serialize blocks to bitstream
        for block in &block_coeff {
            for sq in block {
                let num_zeroes = tree.get_code(sq.num_zeroes);
                let num_bits = tree.get_code(sq.coeff_size);

                debug_assert!(num_zeroes.len > 0 && num_bits.len > 0);

                bitwriter.write(num_zeroes.len, num_zeroes.val)?;
                bitwriter.write(num_bits.len, num_bits.val)?;

                if sq.coeff_size > 0 {
                    bitwriter.write_signed(sq.coeff_size as u32, sq.coeff)?;
                }
            }
        }

        // flush any partial bytes
        bitwriter.byte_align()?;

        // retrieve packet payload bytes
        let packet_data = packet_data.into_inner();

        // write packet header + data

        writer.write_u8(1)?; // packet type = iframe
        writer.write_u32::<LittleEndian>(packet_data.len() as u32)?;
        writer.write_all(&packet_data)?;

        Ok(())
    }

    fn write_pframe_packet(f: &EncodedPFrame, writer: &mut W) -> Result<(), std::io::Error> {
        // serialize packet data
        let mut packet_data = Cursor::new(Vec::new());
        let mut bitwriter = BitWriter::endian(&mut packet_data, bitstream_io::LittleEndian);

        // gather RLE-encoded block coefficients for each plane
        let mut block_coeff = Vec::new();
        let mut symbol_table = [0;16];

        for b in &f.y.blocks {
            match b.subblocks {
                Some(subblocks) => {
                    let mut coeff = Vec::new();
                    coeff.extend_from_slice(&subblocks[0].m);
                    coeff.extend_from_slice(&subblocks[1].m);
                    coeff.extend_from_slice(&subblocks[2].m);
                    coeff.extend_from_slice(&subblocks[3].m);
                    let mut rle_sequence = Vec::new();
                    rle_encode(&mut rle_sequence, &coeff);
                    update_table(&mut symbol_table, &rle_sequence);

                    block_coeff.push(rle_sequence);
                }
                None => {
                }
            }
        }

        for b in &f.u.blocks {
            match b.subblocks {
                Some(subblocks) => {
                    let mut coeff = Vec::new();
                    coeff.extend_from_slice(&subblocks[0].m);
                    coeff.extend_from_slice(&subblocks[1].m);
                    coeff.extend_from_slice(&subblocks[2].m);
                    coeff.extend_from_slice(&subblocks[3].m);
                    let mut rle_sequence = Vec::new();
                    rle_encode(&mut rle_sequence, &coeff);
                    update_table(&mut symbol_table, &rle_sequence);

                    block_coeff.push(rle_sequence);
                }
                None => {
                }
            }
        }

        for b in &f.v.blocks {
            match b.subblocks {
                Some(subblocks) => {
                    let mut coeff = Vec::new();
                    coeff.extend_from_slice(&subblocks[0].m);
                    coeff.extend_from_slice(&subblocks[1].m);
                    coeff.extend_from_slice(&subblocks[2].m);
                    coeff.extend_from_slice(&subblocks[3].m);
                    let mut rle_sequence = Vec::new();
                    rle_encode(&mut rle_sequence, &coeff);
                    update_table(&mut symbol_table, &rle_sequence);

                    block_coeff.push(rle_sequence);
                }
                None => {
                }
            }
        }

        // create huffman tree for encoding RLE results
        let tree = rle_create_huffman(&symbol_table);
        let tree_table = tree.get_table();

        // write symbol frequency table
        for i in 0..16 {
            bitwriter.write(8, tree_table[i] as u8)?;
        }

        // we currently create four qtables: two for i-frames (0, 1) and two for p-frames (2, 3)
        // note: (one qtable index per plane)
        bitwriter.write(8, 2_u8)?;
        bitwriter.write(8, 3_u8)?;
        bitwriter.write(8, 3_u8)?;

        // write block headers
        for b in &f.y.blocks {
            let has_mvec = b.motion_x != 0 || b.motion_y != 0;

            bitwriter.write_bit(has_mvec)?;
            bitwriter.write_bit(b.subblocks.is_some())?;

            if has_mvec {
                bitwriter.write_signed(7, b.motion_x as i32)?;
                bitwriter.write_signed(7, b.motion_y as i32)?;
            }
        }

        for b in &f.u.blocks {
            let has_mvec = b.motion_x != 0 || b.motion_y != 0;

            bitwriter.write_bit(has_mvec)?;
            bitwriter.write_bit(b.subblocks.is_some())?;

            if has_mvec {
                bitwriter.write_signed(7, b.motion_x as i32)?;
                bitwriter.write_signed(7, b.motion_y as i32)?;
            }
        }

        for b in &f.v.blocks {
            let has_mvec = b.motion_x != 0 || b.motion_y != 0;

            bitwriter.write_bit(has_mvec)?;
            bitwriter.write_bit(b.subblocks.is_some())?;

            if has_mvec {
                bitwriter.write_signed(7, b.motion_x as i32)?;
                bitwriter.write_signed(7, b.motion_y as i32)?;

                assert!(b.motion_x >= -16 && b.motion_x <= 16);
                assert!(b.motion_y >= -16 && b.motion_y <= 16);
            }
        }

        // serialize block data to bitstream
        for block in &block_coeff {
            for sq in block {
                let num_zeroes = tree.get_code(sq.num_zeroes);
                let num_bits = tree.get_code(sq.coeff_size);

                bitwriter.write(num_zeroes.len, num_zeroes.val)?;
                bitwriter.write(num_bits.len, num_bits.val)?;

                if sq.coeff_size > 0 {
                    bitwriter.write_signed(sq.coeff_size as u32, sq.coeff)?;
                }
            }
        }

        // flush any partial bytes
        bitwriter.byte_align()?;

        // retrieve packet payload bytes
        let packet_data = packet_data.into_inner();

        // write packet header + data

        writer.write_u8(2)?; // packet type = pframe
        writer.write_u32::<LittleEndian>(packet_data.len() as u32)?;
        writer.write_all(&packet_data)?;

        Ok(())
    }
}