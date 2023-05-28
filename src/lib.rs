pub mod plane;
pub mod frame;
pub mod enc;
pub mod dec;

mod dct;
mod common;
mod huffman;
mod rle;
mod qoa;

#[cfg(test)]
mod tests {
    use std::{path::Path, fs::{File, self}, io::{Cursor, Seek, Read}, time::Instant, hint::black_box};

    use bitstream_io::{BitWriter, BitWrite, BitReader, BitRead};
    use byteorder::{ReadBytesExt, LittleEndian};
    use image::{io::Reader as ImageReader, RgbImage};

    use crate::{dct::*, frame::VideoFrame, plane::VideoPlane, enc::Encoder, dec::Decoder, rle};

    #[test]
    fn test_dct() {
        let data: [f32;8] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];

        let mut dct = data;
        DctMatrix8x8::fast_dct8_transform(&mut dct);

        println!("DCT: {:?}", dct);

        let mut out = dct;
        DctMatrix8x8::fast_dct8_inverse_transform(&mut out);

        println!("Output: {:?}", out);
    }

    #[test]
    fn test_dct_encode() {
        let mut dct = DctMatrix8x8::new();

        for i in 0..64 {
            dct.m[i] = (i * 4) as f32 - 128.0;
        }

        println!("Input: {:?}", dct);

        dct.dct_transform_rows();
        dct.dct_transform_columns();

        let qdct = dct.encode(&Q_TABLE_INTRA);
        println!("Quantized: {:?}", qdct);

        let mut dct2 = DctMatrix8x8::decode(&qdct, &Q_TABLE_INTRA);

        dct2.dct_inverse_transform_columns();
        dct2.dct_inverse_transform_rows();

        println!("Output: {:?}", dct2);
    }

    #[test]
    fn test_entropy() {
        let test_data = [10, 0, 0, 5, 3, 0, 0, 0, 0, -10];
        let mut rle_sequence = Vec::new();
        rle::rle_encode(&mut rle_sequence, &test_data);

        let mut table = [0;16];
        rle::update_table(&mut table, &rle_sequence);

        let tree = rle::rle_create_huffman(&table);
        let mut tmp_buf = Cursor::new(Vec::new());
        let mut bitwriter = BitWriter::endian(&mut tmp_buf, bitstream_io::LittleEndian);

        for sq in &rle_sequence {
            let num_zeroes = tree.get_code(sq.num_zeroes);
            let num_bits = tree.get_code(sq.coeff_size);

            assert!(num_zeroes.len > 0 && num_bits.len > 0);

            bitwriter.write(num_zeroes.len, num_zeroes.val).unwrap();
            bitwriter.write(num_bits.len, num_bits.val).unwrap();

            if sq.coeff_size > 0 {
                bitwriter.write_signed(sq.coeff_size as u32, sq.coeff).unwrap();
            }
        }

        bitwriter.byte_align().unwrap();

        let rle_coded = tmp_buf.into_inner();

        println!("Test data encoded to {} bytes", rle_coded.len());

        let mut rle_reader = Cursor::new(rle_coded);
        let mut bitreader = BitReader::endian(&mut rle_reader, bitstream_io::LittleEndian);

        let total_bits = bitreader.seek_bits(std::io::SeekFrom::End(0)).unwrap();
        bitreader.seek_bits(std::io::SeekFrom::Start(0)).unwrap();

        let mut out_data = [0;10];

        let mut out_idx = 0;
        while out_idx < out_data.len() {
            let num_zeroes = tree.read(&mut bitreader, total_bits).unwrap() as usize;
            out_idx += num_zeroes;

            let num_bits = tree.read(&mut bitreader, total_bits).unwrap();

            // if num_bits is 0, then this is only a run of 0s with no value
            if num_bits > 0 {
                let coeff = bitreader.read_signed::<i16>(num_bits as u32).unwrap();
                out_data[out_idx] = coeff;

                out_idx += 1;
            }
        }

        println!("RLE decoded to: {:?}", out_data);

        test_data.iter().zip(out_data).for_each(|(a, b)| {
            assert!(*a == b);
        });
    }

    #[test]
    fn test_entropy_2() {
        let mut infile = File::open("test_coeff.bin").unwrap();
        let infile_len = infile.seek(std::io::SeekFrom::End(0)).unwrap() as usize;
        infile.seek(std::io::SeekFrom::Start(0)).unwrap();

        let mut test_data = vec![0;infile_len / 2];

        for i in 0..test_data.len() {
            test_data[i] = infile.read_i16::<LittleEndian>().unwrap();
        }

        let mut rle_sequence = Vec::new();
        rle::rle_encode(&mut rle_sequence, &test_data);

        let mut table = [0;16];
        rle::update_table(&mut table, &rle_sequence);

        let tree = rle::rle_create_huffman(&table);
        let mut tmp_buf = Cursor::new(Vec::new());
        let mut bitwriter = BitWriter::endian(&mut tmp_buf, bitstream_io::LittleEndian);

        let mut bits_written = 0;

        for sq in &rle_sequence {
            let num_zeroes = tree.get_code(sq.num_zeroes);
            let num_bits = tree.get_code(sq.coeff_size);

            assert!(num_zeroes.len > 0 && num_bits.len > 0);

            bitwriter.write(num_zeroes.len, num_zeroes.val).unwrap();
            bitwriter.write(num_bits.len, num_bits.val).unwrap();

            if sq.coeff_size > 0 {
                bitwriter.write_signed(sq.coeff_size as u32, sq.coeff).unwrap();
            }

            bits_written += num_zeroes.len + num_bits.len + sq.coeff_size as u32;
        }

        bitwriter.byte_align().unwrap();

        let rle_coded = tmp_buf.into_inner();

        println!("Test data encoded ({} bytes -> {} bytes, {} bits)", infile_len, rle_coded.len(), bits_written);

        let mut rle_reader = Cursor::new(rle_coded);
        let mut bitreader = BitReader::endian(&mut rle_reader, bitstream_io::LittleEndian);

        let total_bits = bitreader.seek_bits(std::io::SeekFrom::End(0)).unwrap();
        bitreader.seek_bits(std::io::SeekFrom::Start(0)).unwrap();

        let mut out_data = vec![0;test_data.len()];

        let mut out_idx = 0;
        let mut run_idx = 0;
        while out_idx < out_data.len() {
            let num_zeroes = tree.read(&mut bitreader, total_bits).unwrap() as usize;
            out_idx += num_zeroes;

            let num_bits = tree.read(&mut bitreader, total_bits).unwrap();

            let run = &rle_sequence[run_idx];
            assert!(run.num_zeroes == num_zeroes as u8);
            assert!(run.coeff_size == num_bits);

            // if num_bits is 0, then this is only a run of 0s with no value
            if num_bits > 0 {
                let coeff = bitreader.read_signed::<i16>(num_bits as u32).unwrap();
                out_data[out_idx] = coeff;
                out_idx += 1;
            }

            run_idx += 1;
        }

        test_data.iter().zip(out_data).for_each(|(a, b)| {
            assert!(*a == b);
        });
    }

    #[test]
    fn test_encode_1() {
        let test_frame = load_frame("test1.png");
        let mut encoder = Encoder::new(test_frame.width, test_frame.height, 30, 44100, 2, 5, 6);
        encoder.encode_iframe(&test_frame);
        encoder.encode_pframe(&test_frame);

        let mut outfile = File::create("test.pfv").unwrap();
        encoder.write(&mut outfile).unwrap();

        println!("File written");
    }

    #[test]
    fn test_decode_1() {
        let infile = File::open("test.pfv").unwrap();
        let mut decoder = Decoder::new(infile, 6).unwrap();

        let mut outframe = 0;

        while decoder.advance_frame(&mut |frame| {
            // write video frame to file
            let frame_out_path = format!("test_frames_out/{:0>3}.png", outframe);
            save_frame(frame_out_path, frame);
            outframe += 1;
        }, &mut |_| {}).unwrap() {}

        println!("Decoded {} frames", outframe);
    }

    #[test]
    fn test_encode_2() {
        let mut inp_audio_file = File::open("test_audio.wav").unwrap();
        let (audio_header, audio_data) = wav::read(&mut inp_audio_file).unwrap();

        let mut encoder = Encoder::new(512, 384, 30, audio_header.sampling_rate, audio_header.channel_count as u32, 2, 6);

        for frame_id in 1..162 {
            let frame_path = format!("test_frames/{:0>3}.png", frame_id);
            let frame = load_frame(frame_path);

            if (frame_id - 1) % 16 == 0 {
                encoder.encode_iframe(&frame);
            } else {
                encoder.encode_pframe(&frame);
            }

            println!("Encoded: {} / {}", frame_id, 162);
        }

        let audio_data: Vec<i16> = match audio_data {
            wav::BitDepth::Eight(v) => {
                v.iter().map(|x| {
                    let f = (*x as f32 / 128.0) - 1.0;
                    (f * 32768.0) as i16
                }).collect()
            }
            wav::BitDepth::Sixteen(v) => {
                v
            }
            wav::BitDepth::ThirtyTwoFloat(v) => {
                v.iter().map(|x| {
                    (*x * 32768.0) as i16
                }).collect()
            }
            _ => {
                panic!("Not implemented")
            }
        };

        encoder.append_audio(&audio_data);

        let mut outfile = File::create("test2.pfv").unwrap();
        encoder.write(&mut outfile).unwrap();

        println!("File written");
    }

    #[test]
    fn test_decode_2() {
        let infile = File::open("test2.pfv").unwrap();
        let mut decoder = Decoder::new(infile, 6).unwrap();

        let channels = decoder.channels();

        let mut outframe = 0;
 
        while decoder.advance_frame(&mut |frame| {
            // write video frame to file
            let frame_out_path = format!("test_frames_out_2/{:0>3}.png", outframe);
            save_frame(frame_out_path, frame);
            outframe += 1;
            println!("Decoded {}", outframe);
        }, &mut |audio| {
            println!("Decoded audio ({} samples)", audio.len() / channels as usize);
        }).unwrap() {}
    }

    #[test]
    fn test_decode_speed_2() {
        for run in 0..50 {
            println!("RUN {}", run);

            let mut infile = File::open("test2.pfv").unwrap();
            let mut filebuf = Vec::new();
            infile.read_to_end(&mut filebuf).unwrap();

            let infile = Cursor::new(filebuf);

            let mut decoder = Decoder::new(infile, 6).unwrap();

            let mut outframe = 0;

            let start = Instant::now();

            while decoder.advance_frame(&mut |frame| {
                outframe += 1;
                black_box(frame);
            }, &mut |_| {}).unwrap() {}

            let duration = start.elapsed().as_millis();
            println!("Decoded {} frames in {} ms", outframe, duration);
        }
    }

    fn load_frame<Q: AsRef<Path>>(path: Q) -> VideoFrame {
        let src_img = ImageReader::open(path).unwrap().decode().unwrap().into_rgb8();
        
        let yuv_pixels: Vec<[u8;3]> = src_img.pixels().map(|rgb| {
            // https://en.wikipedia.org/wiki/YCbCr - "JPEG Conversion"
            let y = (0.299 * rgb.0[0] as f32) + (0.587 * rgb.0[1] as f32) + (0.114 * rgb.0[2] as f32);
            let u = 128.0 - (0.168736 * rgb.0[0] as f32) - (0.331264 * rgb.0[1] as f32) + (0.5 * rgb.0[2] as f32);
            let v = 128.0 + (0.5 * rgb.0[0] as f32) - (0.418688 * rgb.0[1] as f32) - (0.081312 * rgb.0[2] as f32);
            [y as u8, u as u8, v as u8]
        }).collect();

        // split into three planes
        let y_buffer: Vec<_> = yuv_pixels.iter().map(|x| x[0]).collect();
        let u_buffer: Vec<_> = yuv_pixels.iter().map(|x| x[1]).collect();
        let v_buffer: Vec<_> = yuv_pixels.iter().map(|x| x[2]).collect();

        let y_plane = VideoPlane::from_slice(src_img.width() as usize, src_img.height() as usize, &y_buffer);
        let u_plane = VideoPlane::from_slice(src_img.width() as usize, src_img.height() as usize, &u_buffer);
        let v_plane = VideoPlane::from_slice(src_img.width() as usize, src_img.height() as usize, &v_buffer);

        VideoFrame::from_planes(src_img.width() as usize, src_img.height() as usize, y_plane, u_plane, v_plane)
    }

    fn save_frame<Q: AsRef<Path>>(path: Q, frame: &VideoFrame) {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent).unwrap();
        }

        let plane_u = frame.plane_u.double();
        let plane_v = frame.plane_v.double();

        let yuv_pixels: Vec<[u8;3]> = frame.plane_y.pixels.iter().enumerate().map(|(idx, y)| {
            let y = *y;
            let u = plane_u.pixels[idx];
            let v = plane_v.pixels[idx];
            
            [y, u, v]
        }).collect();

        let mut rgb_buf: Vec<u8> = Vec::new();

        for yuv in yuv_pixels.iter() {
            let y = yuv[0] as f32;
            let u = yuv[1] as f32 - 128.0;
            let v = yuv[2] as f32 - 128.0;
            
            // https://en.wikipedia.org/wiki/YCbCr - "JPEG Conversion"
            let r = y + (1.402 * v);
            let g = y - (0.344136 * u) - (0.714136 * v);
            let b = y + (1.772 * u);

            rgb_buf.push(r as u8);
            rgb_buf.push(g as u8);
            rgb_buf.push(b as u8);
        }

        let img_buf = RgbImage::from_vec(frame.width as u32, frame.height as u32, rgb_buf).unwrap();
        img_buf.save(path).unwrap();
    }
}
