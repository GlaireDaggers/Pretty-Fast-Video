## Pretty Fast Video

Toy video codec designed as a successor to [Pretty Good Video](https://github.com/GlaireDaggers/Pretty-Good-Video)

Goals are to improve:

- Quality
- API design
- Codec structure
- (Hopefully) performance

## Usage

### Encoding Video

Create pfv_rs::enc::Encoder, feed in frames & audio, and then write to file:

```rs
use pfv_rs::enc::Encoder;

let mut enc = Encoder::new(width, height, framerate, samplerate, audio_channels, quality, num_threads);

// feed in frames as VideoFrames (1 keyframe every 15 frames)
for (idx, frame) in &my_frames.iter().enumerate() {
  if idx % 15 == 0 {
    enc.encode_iframe(frame);
  } else {
    enc.encode_pframe(frame);
  }
}

// append audio to be encoded (interleaved L/R samples)
enc.append_audio(my_audio);

// write file to disk
let mut out_video = File::create("my_video.pfv").unwrap();
enc.write(&mut out_video).unwrap();
```

### Decoding Video

Create pfv_rs::dec::Decoder and call advance_delta every frame, passing in elapsed time since previous frame, and handling video & audio using closures:

```rs
use pgv_rs::dec::Decoder;

let mut dec = Decoder::new(my_file, num_threads).unwrap();

while dec.advance_delta(delta_time, |frame| {
    // do something with returned &VideoFrame
}, |audio| {
    // do something with returned &[i16]
}).unwrap() {}
```

Alternatively, you may call advance_frame to skip directly to the next frame without passing a delta parameter. The signature is the same.

Both functions will also return Ok(true) if there is more data to read in the file, or Ok(false) if the decoder has reached the end of the file.

## Algorithms

### Video

Video frame encoding is pretty standard as far as video codecs go. Frames are split into 16x16 macroblocks, which are further divided into 8x8 subblocks. Each subblock is DCT transformed & quantized to reduce the number of bits required for storage. Coefficients are further compressed using entropy coding.

There are three kinds of frames: drop frames, i-frames, and p-frames.

- Drop frames encode nothing, and are just treated as being unchanged since the previous frame.
- I-Frames just encode a full frame.
- P-Frames encode a frame as a *delta* from the previous frame. Each macroblock has a pixel offset from the previous frame to copy from, and the macroblock may also encode the per-pixel delta from previous frame (quantized to the 0..255 range).

### Audio

Audio packets are heavily based on the [QOA](https://qoaformat.org/) audio format, due to its ease of implementation, decent audio quality vs compression ratio, and high decoding performance.