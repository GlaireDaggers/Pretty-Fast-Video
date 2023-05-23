extern crate pfv_rs;

use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, Arc};
use std::{fs::File, io::BufReader};

use pfv_rs::dec::Decoder;

use sdl2::audio::{AudioCallback, AudioSpecDesired};
use sdl2::{event::Event, render, pixels::PixelFormatEnum, rect::Rect};
use sdl2::pixels::Color;

struct PFVAudio {
    buffer: Arc<Mutex<Vec<i16>>>,
    audio_buffer_flag: Arc<AtomicBool>,
}

impl AudioCallback for PFVAudio {
    type Channel = i16;

    fn callback(&mut self, out: &mut [i16]) {
        let mut buf = self.buffer.lock().unwrap();
        let samples_to_write = out.len().min(buf.len());

        if samples_to_write < out.len() {
            self.audio_buffer_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        if samples_to_write > 0 {
            out[0..samples_to_write].copy_from_slice(&buf[0..samples_to_write]);
            buf.drain(0..samples_to_write);
        }
    }
}

fn main() {
    let infile = File::open("test2.pfv").unwrap();
    let infile = BufReader::new(infile);

    let mut decoder = Decoder::new(infile, 8).unwrap();
    let channels = decoder.channels();

    let audio_buf: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));
    let audio_buffer_flag = Arc::new(AtomicBool::new(false));

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let timer_subsystem = sdl_context.timer().unwrap();

    let window = video_subsystem.window("PFV Codec Test", decoder.width() as u32, decoder.height() as u32)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().present_vsync().build().unwrap();

    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();

    let desired_spec = {
        AudioSpecDesired {
            freq: Some(decoder.samplerate() as i32),
            channels: Some(decoder.channels() as u8),
            samples: Some(2048)
        }
    };

    let mut bufferlen = 0;

    let device = audio_subsystem.open_playback(None, &desired_spec, |actual_spec| {
        bufferlen = actual_spec.samples as usize * actual_spec.channels as usize;
        PFVAudio {
            audio_buffer_flag: audio_buffer_flag.clone(),
            buffer: audio_buf.clone()
        }
    }).unwrap();

    let tex_creator = canvas.texture_creator();
    let mut tex = {
        tex_creator.create_texture(PixelFormatEnum::IYUV,
            render::TextureAccess::Streaming, decoder.width() as u32, decoder.height() as u32).unwrap()
    };

    let mut event_pump = sdl_context.event_pump().unwrap();

    let mut frametimer = timer_subsystem.performance_counter();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} => {
                    break 'running
                },
                _ => {}
            }
        }

        let new_frametimer = timer_subsystem.performance_counter();
        let delta = (new_frametimer - frametimer) as f32 / timer_subsystem.performance_frequency() as f32;
        frametimer = new_frametimer;

        let playing = decoder.advance_delta(delta as f64, &mut |frame| {
            tex.update_yuv(Rect::new(0, 0, frame.width as u32, frame.height as u32), &frame.plane_y.pixels, frame.width,
                &frame.plane_u.pixels, (frame.width / 2) as usize,
                &frame.plane_v.pixels, (frame.width / 2) as usize).unwrap();
        }, &mut |audio| {
            let mut buf = audio_buf.lock().unwrap();
            buf.extend_from_slice(audio);

            if buf.len() > (bufferlen * channels as usize * 4) {
                device.resume();
            }
        }).unwrap();

        // if audio callback indicates it ran out of audio, pause and allow more audio to buffer before resuming
        if audio_buffer_flag.swap(false, std::sync::atomic::Ordering::Relaxed) {
            device.pause();
        }

        if !playing {
            let mut buf = audio_buf.lock().unwrap();
            buf.clear();
            decoder.reset().unwrap();
        }

        canvas.copy(&tex, None, None).unwrap();
        canvas.present();
    }
}
