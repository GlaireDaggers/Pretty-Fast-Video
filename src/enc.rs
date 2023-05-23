use std::io::{Write, Cursor};

use bitstream_io::{BitWriter, BitWrite};
use byteorder::{WriteBytesExt, LittleEndian};

use crate::common::{EncodedFrame, EncodedIFrame, PFV_MAGIC, PFV_VERSION, EncodedPFrame};
use crate::frame::VideoFrame;
use crate::dct::{Q_TABLE_INTER, Q_TABLE_INTRA};
use crate::plane::VideoPlane;
use crate::rle::{rle_encode, rle_create_huffman};

pub struct Encoder {
    width: usize,
    height: usize,
    framerate: u32,
    samplerate: u32,
    channels: u32,
    prev_frame: VideoFrame,
    px_err: f32,
    qtable_inter: [f32;64],
    qtable_intra: [f32;64],
    frames: Vec<EncodedFrame>,
}

impl Encoder {
    pub fn new(width: usize, height: usize, framerate: u32, samplerate: u32, channels: u32, quality: i32) -> Encoder {
        assert!(quality >= 0 && quality <= 10);

        let qscale = quality as f32 * 0.25;
        let px_err = quality as f32 * 1.5;

        Encoder { width: width, height: height, framerate: framerate, samplerate: samplerate, channels: channels,
            prev_frame: VideoFrame::new_padded(width, height),
            px_err: px_err,
            qtable_inter: Q_TABLE_INTER.map(|x| (x * qscale).max(1.0)),
            qtable_intra: Q_TABLE_INTRA.map(|x| (x * qscale).max(1.0)),
            frames: Vec::new() }
    }

    pub fn encode_iframe(self: &mut Encoder, frame: &VideoFrame) {
        assert!(frame.width == self.width && frame.height == self.height);
        assert!(frame.plane_y.width == frame.width && frame.plane_y.height == frame.height);
        assert!(frame.plane_u.width == frame.width / 2 && frame.plane_u.height == frame.height / 2);
        assert!(frame.plane_v.width == frame.width / 2 && frame.plane_v.height == frame.height / 2);

        let enc_y = frame.plane_y.encode_plane(&self.qtable_intra, 0);
        let dec_y = VideoPlane::decode_plane(&enc_y, &self.qtable_intra);

        let enc_u = frame.plane_u.encode_plane(&self.qtable_intra, 128);
        let dec_u = VideoPlane::decode_plane(&enc_u, &self.qtable_intra);

        let enc_v = frame.plane_v.encode_plane(&self.qtable_intra, 128);
        let dec_v = VideoPlane::decode_plane(&enc_v, &self.qtable_intra);

        self.frames.push(EncodedFrame::IFrame(EncodedIFrame { y: enc_y, u: enc_u, v: enc_v } ));

        self.prev_frame.plane_y.blit(&dec_y, 0, 0, 0, 0, dec_y.width, dec_y.height);
        self.prev_frame.plane_u.blit(&dec_u, 0, 0, 0, 0, dec_u.width, dec_u.height);
        self.prev_frame.plane_v.blit(&dec_v, 0, 0, 0, 0, dec_v.width, dec_v.height);
    }

    pub fn encode_pframe(self: &mut Encoder, frame: &VideoFrame) {
        assert!(frame.width == self.width && frame.height == self.height);
        assert!(frame.plane_y.width == frame.width && frame.plane_y.height == frame.height);
        assert!(frame.plane_u.width == frame.width / 2 && frame.plane_u.height == frame.height / 2);
        assert!(frame.plane_v.width == frame.width / 2 && frame.plane_v.height == frame.height / 2);

        let enc_y = frame.plane_y.encode_plane_delta(&self.prev_frame.plane_y, &self.qtable_inter, self.px_err, 0);
        let dec_y = VideoPlane::decode_plane_delta(&enc_y, &self.prev_frame.plane_y, &self.qtable_inter);

        let enc_u = frame.plane_u.encode_plane_delta(&self.prev_frame.plane_u, &self.qtable_inter, self.px_err, 128);
        let dec_u = VideoPlane::decode_plane_delta(&enc_u, &self.prev_frame.plane_u, &self.qtable_inter);

        let enc_v = frame.plane_v.encode_plane_delta(&self.prev_frame.plane_v, &self.qtable_inter, self.px_err, 128);
        let dec_v = VideoPlane::decode_plane_delta(&enc_v, &self.prev_frame.plane_v, &self.qtable_inter);

        self.frames.push(EncodedFrame::PFrame(EncodedPFrame { y: enc_y, u: enc_u, v: enc_v } ));

        self.prev_frame.plane_y.blit(&dec_y, 0, 0, 0, 0, dec_y.width, dec_y.height);
        self.prev_frame.plane_u.blit(&dec_u, 0, 0, 0, 0, dec_u.width, dec_u.height);
        self.prev_frame.plane_v.blit(&dec_v, 0, 0, 0, 0, dec_v.width, dec_v.height);
    }

    pub fn encode_dropframe(self: &mut Encoder) {
        self.frames.push(EncodedFrame::DropFrame);
    }

    pub fn write<W: Write>(self: &Encoder, writer: &mut W) -> Result<(), std::io::Error> {
        // write PGV header
        writer.write_all(PFV_MAGIC)?;
        writer.write_u32::<LittleEndian>(PFV_VERSION)?;

        writer.write_u16::<LittleEndian>(self.width as u16)?;
        writer.write_u16::<LittleEndian>(self.height as u16)?;
        writer.write_u16::<LittleEndian>(self.framerate as u16)?;

        writer.write_u16::<LittleEndian>(self.samplerate as u16)?;
        writer.write_u16::<LittleEndian>(self.channels as u16)?;

        // write q-tables
        writer.write_u16::<LittleEndian>(2)?;

        for v in self.qtable_intra {
            writer.write_u16::<LittleEndian>(v as u16)?;
        }

        for v in self.qtable_inter {
            writer.write_u16::<LittleEndian>(v as u16)?;
        }

        // write packets to file (interleaved A/V packets)

        for f in &self.frames {
            match f {
                EncodedFrame::IFrame(v) => {
                    Encoder::write_iframe_packet(v, writer)?;
                }
                EncodedFrame::PFrame(v) => {
                    Encoder::write_pframe_packet(v, writer)?;
                }
                EncodedFrame::DropFrame => {
                    Encoder::write_drop_packet(writer)?;
                }
            }
        }

        Ok(())
    }

    fn write_drop_packet<W: Write>(writer: &mut W) -> Result<(), std::io::Error> {
        // write packet header
        writer.write_u8(0)?; // packet type = drop frame
        writer.write_u32::<LittleEndian>(0)?;

        Ok(())
    }

    fn write_iframe_packet<W: Write>(f: &EncodedIFrame, writer: &mut W) -> Result<(), std::io::Error> {
        // serialize packet data
        let mut packet_data = Cursor::new(Vec::new());
        let mut bitwriter = BitWriter::endian(&mut packet_data, bitstream_io::LittleEndian);

        // gather coefficients for each plane into one buffer
        let mut coeff = Vec::new();

        for b in &f.y.blocks {
            coeff.extend_from_slice(&b.subblocks[0].m);
            coeff.extend_from_slice(&b.subblocks[1].m);
            coeff.extend_from_slice(&b.subblocks[2].m);
            coeff.extend_from_slice(&b.subblocks[3].m);
        }

        for b in &f.u.blocks {
            coeff.extend_from_slice(&b.subblocks[0].m);
            coeff.extend_from_slice(&b.subblocks[1].m);
            coeff.extend_from_slice(&b.subblocks[2].m);
            coeff.extend_from_slice(&b.subblocks[3].m);
        }

        for b in &f.v.blocks {
            coeff.extend_from_slice(&b.subblocks[0].m);
            coeff.extend_from_slice(&b.subblocks[1].m);
            coeff.extend_from_slice(&b.subblocks[2].m);
            coeff.extend_from_slice(&b.subblocks[3].m);
        }

        // run length encode all coefficients for this frame
        let mut rle_sequence = Vec::new();
        rle_encode(&mut rle_sequence, &coeff);

        // create huffman tree for encoding RLE results
        let tree = rle_create_huffman(&rle_sequence);
        let tree_table = tree.get_table();

        // write symbol frequency table
        for i in 0..16 {
            bitwriter.write(8, tree_table[i] as u8)?;
        }

        // we currently only create two qtables: one for i-frames (0) and one for p-frames (1)
        // note: (one qtable index per plane)
        bitwriter.write(8, 0_u8)?;
        bitwriter.write(8, 0_u8)?;
        bitwriter.write(8, 0_u8)?;

        // serialize RLE to bitstream
        for sq in &rle_sequence {
            let num_zeroes = tree.get_code(sq.num_zeroes);
            let num_bits = tree.get_code(sq.coeff_size);

            debug_assert!(num_zeroes.len > 0 && num_bits.len > 0);

            bitwriter.write(num_zeroes.len, num_zeroes.val)?;
            bitwriter.write(num_bits.len, num_bits.val)?;

            if sq.coeff_size > 0 {
                bitwriter.write_signed(sq.coeff_size as u32, sq.coeff)?;
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

    fn write_pframe_packet<W: Write>(f: &EncodedPFrame, writer: &mut W) -> Result<(), std::io::Error> {
        // serialize packet data
        let mut packet_data = Cursor::new(Vec::new());
        let mut bitwriter = BitWriter::endian(&mut packet_data, bitstream_io::LittleEndian);

        // gather coefficients for each plane into one buffer
        let mut coeff = Vec::new();

        for b in &f.y.blocks {
            match b.subblocks {
                Some(subblocks) => {
                    coeff.extend_from_slice(&subblocks[0].m);
                    coeff.extend_from_slice(&subblocks[1].m);
                    coeff.extend_from_slice(&subblocks[2].m);
                    coeff.extend_from_slice(&subblocks[3].m);
                }
                None => {
                }
            }
        }

        for b in &f.u.blocks {
            match b.subblocks {
                Some(subblocks) => {
                    coeff.extend_from_slice(&subblocks[0].m);
                    coeff.extend_from_slice(&subblocks[1].m);
                    coeff.extend_from_slice(&subblocks[2].m);
                    coeff.extend_from_slice(&subblocks[3].m);
                }
                None => {
                }
            }
        }

        for b in &f.v.blocks {
            match b.subblocks {
                Some(subblocks) => {
                    coeff.extend_from_slice(&subblocks[0].m);
                    coeff.extend_from_slice(&subblocks[1].m);
                    coeff.extend_from_slice(&subblocks[2].m);
                    coeff.extend_from_slice(&subblocks[3].m);
                }
                None => {
                }
            }
        }

        // run length encode all coefficients for this frame
        let mut rle_sequence = Vec::new();
        rle_encode(&mut rle_sequence, &coeff);

        // create huffman tree for encoding RLE results
        let tree = rle_create_huffman(&rle_sequence);
        let tree_table = tree.get_table();

        // write symbol frequency table
        for i in 0..16 {
            bitwriter.write(8, tree_table[i] as u8)?;
        }

        // we currently only create two qtables: one for i-frames (0) and one for p-frames (1)
        // note: (one qtable index per plane)
        bitwriter.write(8, 1_u8)?;
        bitwriter.write(8, 1_u8)?;
        bitwriter.write(8, 1_u8)?;

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

        // serialize RLE to bitstream
        for sq in &rle_sequence {
            let num_zeroes = tree.get_code(sq.num_zeroes);
            let num_bits = tree.get_code(sq.coeff_size);

            bitwriter.write(num_zeroes.len, num_zeroes.val)?;
            bitwriter.write(num_bits.len, num_bits.val)?;

            if sq.coeff_size > 0 {
                bitwriter.write_signed(sq.coeff_size as u32, sq.coeff)?;
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