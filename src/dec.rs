use std::{io::{Read, Seek, Cursor}, slice::{ChunksExact, Iter}};

use bitstream_io::{BitReader, BitRead};
use byteorder::{ReadBytesExt, LittleEndian};

use crate::{common::{PFV_MAGIC, PFV_VERSION, EncodedMacroBlock, EncodedIPlane, DeltaEncodedMacroBlock, EncodedPPlane}, huffman::{HuffmanTree, HuffmanError}, frame::VideoFrame, plane::VideoPlane, dct::{DctQuantizedMatrix8x8}};

#[derive(Debug, Clone, Copy)]
struct DeltaBlockHeader {
    mvec_x: i8,
    mvec_y: i8,
    has_coeff: bool,
}

pub struct Decoder<TReader: Read + Seek> {
    reader: TReader,
    width: usize,
    height: usize,
    framerate: u32,
    qtables: Vec<[i32;64]>,
    framebuffer: VideoFrame,
    retframe: VideoFrame,
    delta_accum: f64,
    eof: bool,
    reset_pos: u64,
    #[cfg(feature = "multithreading")]
    threadpool: rayon::ThreadPool
}

#[derive(Debug)]
pub enum DecodeError {
    FormatError,
    VersionError,
    IOError(std::io::Error)
}

impl<TReader: Read + Seek> Decoder<TReader> {
    pub fn new(mut reader: TReader, #[cfg(feature = "multithreading")] num_threads: usize) -> Result<Decoder<TReader>, DecodeError> {
        // read header
        let mut magic = [0;8];
        match reader.read_exact(&mut magic) {
            Ok(_) => {}
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let magic_match = magic.iter().zip(PFV_MAGIC.iter()).all(|(a, b)| *a == *b);

        if !magic_match {
            return Err(DecodeError::FormatError);
        }

        // read version
        match reader.read_u32::<LittleEndian>() {
            Ok(ver) => {
                if ver != PFV_VERSION {
                    return Err(DecodeError::VersionError);
                }

                ver
            }
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let width = match reader.read_u16::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let height = match reader.read_u16::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let framerate = match reader.read_u16::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let num_qtable = match reader.read_u16::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        let mut qtables = Vec::new();

        for _ in 0..num_qtable {
            let mut qtable = [0;64];

            for i in 0..64 {
                qtable[i] = match reader.read_u16::<LittleEndian>() {
                    Ok(v) => v as i32,
                    Err(e) => {
                        return Err(DecodeError::IOError(e));
                    }
                };
            }

            qtables.push(qtable);
        }

        let reset_pos = match reader.stream_position() {
            Ok(v) => v,
            Err(e) => {
                return Err(DecodeError::IOError(e));
            }
        };

        #[cfg(feature = "multithreading")]
        {
            Ok(Decoder { reader: reader, width: width as usize, height: height as usize, framerate: framerate as u32,
                qtables: qtables, framebuffer: VideoFrame::new_padded(width as usize, height as usize),
                retframe: VideoFrame::new(width as usize, height as usize), delta_accum: 0.0, eof: false, reset_pos: reset_pos,
                threadpool: rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap() })
        }

        #[cfg(not(feature = "multithreading"))]
        {
            Ok(Decoder { reader: reader, width: width as usize, height: height as usize, framerate: framerate as u32,
                qtables: qtables, framebuffer: VideoFrame::new_padded(width as usize, height as usize),
                retframe: VideoFrame::new(width as usize, height as usize), delta_accum: 0.0, eof: false, reset_pos: reset_pos, })
        }
    }

    pub fn width(self: &Decoder<TReader>) -> usize {
        return self.width;
    }

    pub fn height(self: &Decoder<TReader>) -> usize {
        return self.height;
    }

    pub fn framerate(self: &Decoder<TReader>) -> u32 {
        return self.framerate;
    }

    pub fn reset(self: &mut Decoder<TReader>) -> Result<(), std::io::Error> {
        self.eof = false;
        self.reader.seek(std::io::SeekFrom::Start(self.reset_pos))?;
        Ok(())
    }

    pub fn advance_delta<FV>(self: &mut Decoder<TReader>, delta: f64, onvideo: &mut FV) -> Result<bool, std::io::Error>  where
        FV: FnMut(&VideoFrame) {
        self.delta_accum += delta;
        let delta_per_frame = 1.0 / self.framerate as f64;

        while self.delta_accum >= delta_per_frame {
            if self.advance_frame(onvideo)? == false {
                return Ok(false);
            }
            self.delta_accum -= delta_per_frame;
        }

        Ok(true)
    }

    pub fn advance_frame<FV>(self: &mut Decoder<TReader>, onvideo: &mut FV) -> Result<bool, std::io::Error> where
        FV: FnMut(&VideoFrame) {
        if self.eof {
            return Ok(false);
        }

        loop {
            // read next packet header
            // if we hit EOF, return false

            let packet_type = self.reader.read_u8()?;
            let packet_len = self.reader.read_u32::<LittleEndian>()?;

            match packet_type {
                0 => {
                    // EOF marker
                    self.eof = true;
                    return Ok(false);
                }
                1 => {
                    // iframe. if payload length is zero, this is a drop frame (do nothing)
                    if packet_len > 0 {
                        let mut data = vec![0;packet_len as usize];
                        self.reader.read_exact(&mut data)?;
                        self.decode_iframe(&data)?;

                        self.retframe.plane_y.blit(&self.framebuffer.plane_y, 0, 0, 0, 0, self.retframe.plane_y.width, self.retframe.plane_y.height);
                        self.retframe.plane_u.blit(&self.framebuffer.plane_u, 0, 0, 0, 0, self.retframe.plane_u.width, self.retframe.plane_u.height);
                        self.retframe.plane_v.blit(&self.framebuffer.plane_v, 0, 0, 0, 0, self.retframe.plane_v.width, self.retframe.plane_v.height);

                        onvideo(&self.retframe);
                    }
                    break;
                }
                2 => {
                    // pframe
                    let mut data = vec![0;packet_len as usize];
                    self.reader.read_exact(&mut data)?;
                    self.decode_pframe(&data)?;

                    self.retframe.plane_y.blit(&self.framebuffer.plane_y, 0, 0, 0, 0, self.retframe.plane_y.width, self.retframe.plane_y.height);
                    self.retframe.plane_u.blit(&self.framebuffer.plane_u, 0, 0, 0, 0, self.retframe.plane_u.width, self.retframe.plane_u.height);
                    self.retframe.plane_v.blit(&self.framebuffer.plane_v, 0, 0, 0, 0, self.retframe.plane_v.width, self.retframe.plane_v.height);

                    onvideo(&self.retframe);
                    break;
                }
                _ => {
                    // unrecognized packet type, just skip over packet payload
                    self.reader.seek(std::io::SeekFrom::Current(packet_len as i64))?;
                }
            }
        }

        Ok(true)
    }

    fn decode_iframe(self: &mut Decoder<TReader>, payload: &[u8]) -> Result<(), std::io::Error> {
        let reader = Cursor::new(payload);
        let mut bitreader = BitReader::endian(reader, bitstream_io::LittleEndian);

        let bitstream_length = bitreader.seek_bits(std::io::SeekFrom::End(0))?;
        bitreader.seek_bits(std::io::SeekFrom::Start(0))?;

        // read symbol frequency table
        let mut table = [0;16];

        for i in 0..16 {
            table[i] = bitreader.read::<u8>(8).unwrap();
        }

        // construct huffman tree
        let tree = HuffmanTree::from_table(&table);

        // fetch qtables
        let qtable_y = &self.qtables[bitreader.read::<u8>(8).unwrap() as usize];
        let qtable_u = &self.qtables[bitreader.read::<u8>(8).unwrap() as usize];
        let qtable_v = &self.qtables[bitreader.read::<u8>(8).unwrap() as usize];

        // decode RLE coefficients
        let blocks_wide = self.framebuffer.plane_y.width / 16;
        let blocks_high = self.framebuffer.plane_y.height / 16;

        let chroma_blocks_wide = self.framebuffer.plane_u.width / 16;
        let chroma_blocks_high = self.framebuffer.plane_u.height / 16;

        let total_blocks = (blocks_wide * blocks_high) + (chroma_blocks_wide * chroma_blocks_high * 2);
        let total_subblocks = total_blocks * 4;

        let mut coefficients = vec![0;total_subblocks * 64 as usize];

        let mut out_idx = 0;
        while out_idx < coefficients.len() {
            let num_zeroes = match tree.read(&mut bitreader, bitstream_length) {
                Ok(v) => v,
                Err(e) => match e {
                    HuffmanError::DecodeError => unreachable!(),
                    HuffmanError::IOError(e2) => {
                        return Err(e2);
                    },
                }
            } as usize;

            out_idx += num_zeroes;

            let num_bits = match tree.read(&mut bitreader, bitstream_length) {
                Ok(v) => v,
                Err(e) => match e {
                    HuffmanError::DecodeError => unreachable!(),
                    HuffmanError::IOError(e2) => {
                        return Err(e2);
                    },
                }
            };

            // if num_bits is 0, then this is only a run of 0s with no value
            if num_bits > 0 {
                let coeff = match bitreader.read_signed::<i16>(num_bits as u32) {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(e);
                    }
                };
                coefficients[out_idx] = coeff;

                out_idx += 1;
            }
        }

        let mut subblocks = coefficients.chunks_exact(64);

        // deserialize each plane
        #[cfg(feature = "multithreading")]
        {
            Decoder::<TReader>::deserialize_plane(self.framebuffer.plane_y.width, self.framebuffer.plane_y.height,
                &mut subblocks, qtable_y, &mut self.framebuffer.plane_y, &self.threadpool);
                
            Decoder::<TReader>::deserialize_plane(self.framebuffer.plane_u.width, self.framebuffer.plane_u.height,
                &mut subblocks, qtable_u, &mut self.framebuffer.plane_u, &self.threadpool);
                
            Decoder::<TReader>::deserialize_plane(self.framebuffer.plane_v.width, self.framebuffer.plane_v.height,
                &mut subblocks, qtable_v, &mut self.framebuffer.plane_v, &self.threadpool);
        }

        #[cfg(not(feature = "multithreading"))]
        {
            Decoder::<TReader>::deserialize_plane(self.framebuffer.plane_y.width, self.framebuffer.plane_y.height,
                &mut subblocks, qtable_y, &mut self.framebuffer.plane_y);
                
            Decoder::<TReader>::deserialize_plane(self.framebuffer.plane_u.width, self.framebuffer.plane_u.height,
                &mut subblocks, qtable_u, &mut self.framebuffer.plane_u);
                
            Decoder::<TReader>::deserialize_plane(self.framebuffer.plane_v.width, self.framebuffer.plane_v.height,
                &mut subblocks, qtable_v, &mut self.framebuffer.plane_v);
        }

        Ok(())
    }

    fn decode_pframe(self: &mut Decoder<TReader>, payload: &[u8]) -> Result<(), std::io::Error> {
        let reader = Cursor::new(payload);
        let mut bitreader = BitReader::endian(reader, bitstream_io::LittleEndian);

        let bitstream_length = bitreader.seek_bits(std::io::SeekFrom::End(0))?;
        bitreader.seek_bits(std::io::SeekFrom::Start(0))?;

        // read symbol frequency table
        let mut table = [0;16];

        for i in 0..16 {
            table[i] = bitreader.read::<u8>(8).unwrap();
        }

        // construct huffman tree
        let tree = HuffmanTree::from_table(&table);

        // fetch qtables
        let qtable_y = &self.qtables[bitreader.read::<u8>(8)? as usize];
        let qtable_u = &self.qtables[bitreader.read::<u8>(8)? as usize];
        let qtable_v = &self.qtables[bitreader.read::<u8>(8)? as usize];

        // read block headers
        let blocks_wide = self.framebuffer.plane_y.width / 16;
        let blocks_high = self.framebuffer.plane_y.height / 16;

        let chroma_blocks_wide = self.framebuffer.plane_u.width / 16;
        let chroma_blocks_high = self.framebuffer.plane_u.height / 16;

        let total_blocks = (blocks_wide * blocks_high) + (chroma_blocks_wide * chroma_blocks_high * 2);

        let mut block_headers = Vec::with_capacity(total_blocks);

        for _ in 0..total_blocks {
            let mut header = DeltaBlockHeader { mvec_x: 0, mvec_y: 0, has_coeff: false };
            let has_mvec = bitreader.read_bit()?;
            header.has_coeff = bitreader.read_bit()?;

            if has_mvec {
                header.mvec_x = bitreader.read_signed(7)?;
                header.mvec_y = bitreader.read_signed(7)?;
            }

            block_headers.push(header);
        }

        // decode block coefficients

        let mut coefficients = vec![0;total_blocks * 256];

        for (idx, header) in block_headers.iter().enumerate() {
            let mut block_coeff = [0;256];
            let block_offset = idx * 256;
            if header.has_coeff {
                // read 256 coefficients from bit stream
                let mut out_idx = 0;
                while out_idx < 256 {
                    let num_zeroes = match tree.read(&mut bitreader, bitstream_length) {
                        Ok(v) => v,
                        Err(e) => match e {
                            HuffmanError::DecodeError => unreachable!(),
                            HuffmanError::IOError(e2) => {
                                return Err(e2);
                            },
                        }
                    } as usize;

                    out_idx += num_zeroes;

                    let num_bits = match tree.read(&mut bitreader, bitstream_length) {
                        Ok(v) => v,
                        Err(e) => match e {
                            HuffmanError::DecodeError => unreachable!(),
                            HuffmanError::IOError(e2) => {
                                return Err(e2);
                            },
                        }
                    };

                    // if num_bits is 0, then this is only a run of 0s with no value
                    if num_bits > 0 {
                        let coeff = bitreader.read_signed::<i16>(num_bits as u32)?;
                        block_coeff[out_idx] = coeff;

                        out_idx += 1;
                    }
                }
            }
            coefficients[block_offset..block_offset+256].copy_from_slice(&block_coeff);
        }

        let mut subblocks = coefficients.chunks_exact(64);
        let mut headers = block_headers.iter();

        // deserialize each plane
        #[cfg(feature = "multithreading")]
        {
            Decoder::<TReader>::deserialize_plane_delta(self.framebuffer.plane_y.width, self.framebuffer.plane_y.height,
                &mut headers, &mut subblocks, qtable_y, &mut self.framebuffer.plane_y, &self.threadpool);
                
            Decoder::<TReader>::deserialize_plane_delta(self.framebuffer.plane_u.width, self.framebuffer.plane_u.height,
                &mut headers, &mut subblocks, qtable_u, &mut self.framebuffer.plane_u, &self.threadpool);
                
            Decoder::<TReader>::deserialize_plane_delta(self.framebuffer.plane_v.width, self.framebuffer.plane_v.height,
                &mut headers, &mut subblocks, qtable_v, &mut self.framebuffer.plane_v, &self.threadpool);
        }

        #[cfg(not(feature = "multithreading"))]
        {
            Decoder::<TReader>::deserialize_plane_delta(self.framebuffer.plane_y.width, self.framebuffer.plane_y.height,
                &mut headers, &mut subblocks, qtable_y, &mut self.framebuffer.plane_y);
                
            Decoder::<TReader>::deserialize_plane_delta(self.framebuffer.plane_u.width, self.framebuffer.plane_u.height,
                &mut headers, &mut subblocks, qtable_u, &mut self.framebuffer.plane_u);
                
            Decoder::<TReader>::deserialize_plane_delta(self.framebuffer.plane_v.width, self.framebuffer.plane_v.height,
                &mut headers, &mut subblocks, qtable_v, &mut self.framebuffer.plane_v);
        }

        Ok(())
    }

    fn deserialize_plane(width: usize, height: usize, subblocks: &mut ChunksExact<i16>, q_table: &[i32;64], target: &mut VideoPlane, #[cfg(feature = "multithreading")] tp: &rayon::ThreadPool) {
        let blocks_wide = width / 16;
        let blocks_high = height / 16;
        let total_blocks = blocks_wide * blocks_high;

        let mut enc_plane = EncodedIPlane { blocks_wide: blocks_wide, blocks_high: blocks_high, width: width, height: height,
            blocks: Vec::with_capacity(total_blocks) };

        for _ in 0..total_blocks {
            let s0 = subblocks.next().unwrap();
            let s1 = subblocks.next().unwrap();
            let s2 = subblocks.next().unwrap();
            let s3 = subblocks.next().unwrap();

            let block = EncodedMacroBlock { subblocks: [
                DctQuantizedMatrix8x8::from_slice(s0),
                DctQuantizedMatrix8x8::from_slice(s1),
                DctQuantizedMatrix8x8::from_slice(s2),
                DctQuantizedMatrix8x8::from_slice(s3),
            ] };

            enc_plane.blocks.push(block);
        }

        #[cfg(feature = "multithreading")]
        VideoPlane::decode_plane_into(&enc_plane, q_table, target, tp);

        #[cfg(not(feature = "multithreading"))]
        VideoPlane::decode_plane_into(&enc_plane, q_table, target);
    }

    fn deserialize_plane_delta(width: usize, height: usize, headers: &mut Iter<DeltaBlockHeader>, subblocks: &mut ChunksExact<i16>, q_table: &[i32;64], target: &mut VideoPlane,
        #[cfg(feature = "multithreading")] tp: &rayon::ThreadPool) {
        let blocks_wide = width / 16;
        let blocks_high = height / 16;
        let total_blocks = blocks_wide * blocks_high;

        let mut enc_plane = EncodedPPlane { blocks_wide: blocks_wide, blocks_high: blocks_high, width: width, height: height,
            blocks: Vec::with_capacity(total_blocks) };

        for _ in 0..total_blocks {
            let header = headers.next().unwrap();

            let s0 = subblocks.next().unwrap();
            let s1 = subblocks.next().unwrap();
            let s2 = subblocks.next().unwrap();
            let s3 = subblocks.next().unwrap();

            let block = DeltaEncodedMacroBlock {
                motion_x: header.mvec_x,
                motion_y: header.mvec_y,
                subblocks: if header.has_coeff { Some([
                    DctQuantizedMatrix8x8::from_slice(s0),
                    DctQuantizedMatrix8x8::from_slice(s1),
                    DctQuantizedMatrix8x8::from_slice(s2),
                    DctQuantizedMatrix8x8::from_slice(s3),
                ]) } else { None }
            };

            enc_plane.blocks.push(block);
        }

        #[cfg(feature = "multithreading")]
        VideoPlane::decode_plane_delta_into(&enc_plane, target, q_table, tp);

        #[cfg(not(feature = "multithreading"))]
        VideoPlane::decode_plane_delta_into(&enc_plane, target, q_table);
    }
}