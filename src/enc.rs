use std::io::{Write, Cursor};

use bitstream_io::{BitWriter, BitWrite};
use byteorder::{WriteBytesExt, LittleEndian};

use crate::common::{EncodedFrame, EncodedIFrame, PFV_MAGIC, PFV_VERSION, EncodedPFrame};
use crate::frame::VideoFrame;
use crate::dct::{Q_TABLE_INTER, Q_TABLE_INTRA};
use crate::plane::VideoPlane;
use crate::qoa::{LMS, EncodedAudioFrame, QOA_SLICE_LEN, QOA_LMS_LEN, QOA_DEQUANT_TABLE, qoa_lms_predict, qoa_div, QOA_QUANT_TABLE, QOA_FRAME_LEN};
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
    audio_buf: Vec<Vec<i16>>,
}

impl Encoder {
    pub fn new(width: usize, height: usize, framerate: u32, samplerate: u32, channels: u32, quality: i32) -> Encoder {
        assert!(quality >= 0 && quality <= 10);

        let qscale = quality as f32 * 0.25;
        let px_err = quality as f32 * 1.5;

        let mut audio_buf = Vec::new();

        for _ in 0..channels {
            audio_buf.push(Vec::new());
        }

        Encoder { width: width, height: height, framerate: framerate, samplerate: samplerate, channels: channels,
            prev_frame: VideoFrame::new_padded(width, height),
            px_err: px_err,
            qtable_inter: Q_TABLE_INTER.map(|x| (x * qscale).max(1.0)),
            qtable_intra: Q_TABLE_INTRA.map(|x| (x * qscale).max(1.0)),
            frames: Vec::new(),
            audio_buf: audio_buf }
    }

    pub fn append_audio(self: &mut Encoder, audio: &[i16]) {
        // split interleaved audio data into one buffer per channel
        for sample in audio.chunks_exact(self.channels as usize) {
            for ch in 0..self.channels as usize {
                self.audio_buf[ch].push(sample[ch]);
            }
        }
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

    pub fn write<W: Write>(self: &mut Encoder, writer: &mut W) -> Result<(), std::io::Error> {
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

        // write packets to file (interleave A/V packets)

        let samples_per_frame = self.samplerate as f32 / self.framerate as f32;
        let mut sample_accum = 0.0;

        // 250ms audio buffer size (at 44.1khz this means each returned audio chunk will contain 11025 samples)
        let buffer_size = (self.samplerate / 4) as f32;

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

            if self.audio_buf.len() > 0 {
                sample_accum += samples_per_frame;

                while sample_accum >= buffer_size {
                    let samples_to_write = (buffer_size as usize).max(self.audio_buf[0].len());
                    let mut samples: Vec<Vec<i16>> = Vec::new();
                    for buf in &mut self.audio_buf {
                        samples.push(buf.drain(0..samples_to_write).collect());
                    }
                    Encoder::write_audio_packet(&samples, writer)?;
                    sample_accum -= buffer_size;
                }
            }
        }

        Encoder::write_eof(writer)?;

        Ok(())
    }

    // adapted from https://github.com/mattdesl/qoa-format/blob/main/encode.js

    fn encode_audio_frame(audio: &Vec<Vec<i16>>, lmses: &mut Vec<LMS>, sample_offset: usize, frame_len: usize) -> EncodedAudioFrame {
        let mut result = EncodedAudioFrame {
            samples: frame_len as usize,
            slices: Vec::new(),
            lmses: lmses.clone(),
        };

        let mut sample_index = 0;
        while sample_index < frame_len {
            for c in 0..audio.len() {
                let slice_start = sample_index;
                let slice_len = QOA_SLICE_LEN.clamp(0, frame_len - sample_index);

                // brute force search for best scale factor (just loop through all possible scale factors and compare error)

                let mut best_err = i64::MAX as i64;
                let mut best_slice = Vec::new();
                let mut best_slice_scalefactor = 0;
                let mut best_lms = LMS { history: [0;QOA_LMS_LEN], weight: [0;QOA_LMS_LEN] };
                let sampledata = &audio[c];

                for scalefactor in 0..16 {
                    let mut lms = lmses[c];
                    let table = QOA_DEQUANT_TABLE[scalefactor];

                    let mut slice = Vec::new();
                    let mut current_error = 0;
                    let mut idx = slice_start + sample_offset;

                    for _ in 0..slice_len {
                        let sample = sampledata[idx] as i32;
                        idx += 1;

                        let predicted = qoa_lms_predict(lms);
                        let residual = sample - predicted;
                        let scaled = qoa_div(residual, scalefactor);
                        let clamped = scaled.clamp(-8, 8);
                        let quantized = QOA_QUANT_TABLE[(clamped + 8) as usize];
                        let dequantized = table[quantized as usize];
                        let reconstructed = (predicted + dequantized).clamp(i16::MIN as i32, i16::MAX as i32);
                        let error = (sample - reconstructed) as i64;
                        current_error += error * error;
                        if current_error > best_err {
                            break;
                        }

                        lms.update(reconstructed, dequantized);
                        slice.push(quantized);
                    }

                    if current_error < best_err {
                        best_err = current_error;
                        best_slice = slice;
                        best_slice_scalefactor = scalefactor;
                        best_lms = lms;
                    }
                }

                // if best_err is i64::MAX, that implies that *no* suitable scalefactor could be found
                // something has gone wrong here
                assert!(best_err < i64::MAX);

                lmses[c] = best_lms;

                // pack bits into slice - low 4 bits are scale factor, remaining 60 bits are quantized residuals
                let mut slice = (best_slice_scalefactor & 0xF) as u64;

                for i in 0..best_slice.len() {
                    let v = best_slice[i] as u64;
                    slice |= ((v & 0x7) << ((i * 3) + 4)) as u64;
                }

                result.slices.push(slice);
            }

            sample_index += QOA_SLICE_LEN;
        }

        result
    }

    fn write_audio_packet<W: Write>(audio: &Vec<Vec<i16>>, writer: &mut W) -> Result<(), std::io::Error> {
        let samples = audio[0].len();

        assert!(samples < 65536);

        for a in audio {
            assert!(a.len() == samples);
        }

        // init LMS
        let mut lmses: Vec<LMS> = audio.iter().map(|_| {
            LMS {
                weight: [0, 0, -(1 << 13), 1 << 14],
                history: [0, 0, 0, 0]
            }
        }).collect();

        let mut frames = Vec::new();

        let mut sample_index = 0;
        while sample_index < samples {
            let frame_len = QOA_FRAME_LEN.clamp(0, samples - sample_index);
            frames.push(Encoder::encode_audio_frame(&audio, &mut lmses, sample_index, frame_len));
            sample_index += QOA_FRAME_LEN;
        }

        // write total samples in audio packet
        writer.write_u16::<LittleEndian>(samples as u16)?;

        // for each encoded frame:
        //  write number of samples per channel in frame
        //  write total slice count in frame
        //  write LMS history & weights for each channel
        //  write each slice in frame

        for frame in &frames {
            writer.write_u16::<LittleEndian>(frame.samples as u16)?;
            writer.write_u16::<LittleEndian>(frame.slices.len() as u16)?;

            for lms in &frame.lmses {
                for history in lms.history {
                    writer.write_i16::<LittleEndian>(history as i16)?;
                }

                for weight in lms.weight {
                    writer.write_i16::<LittleEndian>(weight as i16)?;
                }
            }

            for slice in &frame.slices {
                writer.write_u64::<LittleEndian>(*slice)?;
            }
        }

        Ok(())
    }

    fn write_eof<W: Write>(writer: &mut W) -> Result<(), std::io::Error> {
        // write packet header
        writer.write_u8(0)?; // packet type = EOF
        writer.write_u32::<LittleEndian>(0)?;

        Ok(())
    }

    fn write_drop_packet<W: Write>(writer: &mut W) -> Result<(), std::io::Error> {
        // write packet header
        writer.write_u8(1)?; // packet type = iframe
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