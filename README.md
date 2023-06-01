## Pretty Fast Video

Minimal video codec designed as a successor to [Pretty Good Video](https://github.com/GlaireDaggers/Pretty-Good-Video)

Goals are to improve:

- Quality
- API design
- Codec structure
- (Hopefully) performance

Current codec version is 2.0.1

## Why?

The goal of PFV is to be to video what QOA is to audio - that is, an extremely minimal codec that aims to prioritize simplicity and decoding speed over compression ratio, is completely free for use, and aimed at playing back video in offline applications such as games.

The primary target to beat for performance has been Theora. While this Rust library does not manage to beat Theora's decoding performance, I have written an accompanying pure C library which *does* beat Theora's decoding performance (especially when scaled to use all available cores in the system, as it is an extremely parallel algorithm): [libpfvdec](https://github.com/GlaireDaggers/libpfvdec)

A future target to explore would be GPU decoding via compute shaders - as each macroblock can already be processed completely in parallel, it should be simple to upload DCT coefficients and perform decoding completely on the GPU, decoding directly into a game-ready texture.

## Usage

### Encoding Video

Create pfv_rs::enc::Encoder, feed in frames, and then write results:

```rs
use pfv_rs::enc::Encoder;

let out_video = File::create("my_video.pfv").unwrap();
let mut enc = Encoder::new(out_video, width, height, framerate, quality, num_threads).unwrap();

// feed in frames as VideoFrames (1 keyframe every 15 frames)
for (idx, frame) in &my_frames.iter().enumerate() {
  if idx % 15 == 0 {
    enc.encode_iframe(frame).unwrap();
  } else {
    enc.encode_pframe(frame).unwrap();
  }
}

// finish PFV stream (will also be automatically called if encoder is dropped)
enc.finish().unwrap();
```

### Decoding Video

Create pfv_rs::dec::Decoder and call advance_delta every frame, passing in elapsed time since previous frame, and handling frames using a closure:

```rs
use pgv_rs::dec::Decoder;

let mut dec = Decoder::new(my_file, num_threads).unwrap();

while dec.advance_delta(delta_time, &mut |frame| {
    // do something with returned &VideoFrame
}).unwrap() {}
```

Alternatively, you may call advance_frame to skip directly to the next frame without passing a delta parameter. The signature is the same.

Both functions will also return Ok(true) if there is more data to read in the file, or Ok(false) if the decoder has reached the end of the file.

## Algorithm Overview

Video frame encoding is pretty standard as far as video codecs go. Frames are split into 16x16 macroblocks, which are further divided into 8x8 subblocks. Each subblock is DCT transformed & quantized to reduce the number of bits required for storage. Coefficients are further compressed using entropy coding.

PFV also employs 4:2:0 chroma subsampling - so U and V chroma planes are half the size of the Y plane on each axis.

There are three kinds of frames: drop frames, i-frames, and p-frames.

- Drop frames encode nothing, and are just treated as being unchanged since the previous frame.
- I-Frames just encode a full frame.
- P-Frames encode a frame as a *delta* from the previous frame. Each macroblock has a pixel offset from the previous frame to copy from, and the macroblock may also encode the per-pixel delta from previous frame (quantized to the 0..255 range).

## Audio

Audio has been removed from the spec as of codec version 2.0.0

You may use any audio stream format of choice with PFV video streams, whether this be embedded in some kind of container format or just shipped alongside
the video files. For lightweight CPU requirements, [QOA](https://qoaformat.org/) is a decent choice for audio tracks.