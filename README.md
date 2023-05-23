## Pretty Fast Video

Toy video codec designed as a successor to [Pretty Good Video](https://github.com/GlaireDaggers/Pretty-Good-Video)

Goals are to improve:

- Quality
- API design
- Codec structure
- (Hopefully) performance

## Algorithms

### Video

Video frame encoding is pretty standard as far as video codecs go. Frames are split into 16x16 macroblocks, which are further divided into 8x8 subblocks. Each subblock is DCT transformed & quantized to reduce the number of bits required for storage. Coefficients are further compressed using entropy coding.

There are three kinds of frames: drop frames, i-frames, and p-frames.

- Drop frames encode nothing, and are just treated as being unchanged since the previous frame.
- I-Frames just encode a full frame.
- P-Frames encode a frame as a *delta* from the previous frame. Each macroblock has a pixel offset from the previous frame to copy from, and the macroblock may also encode the per-pixel delta from previous frame (quantized to the 0..255 range).

### Audio

Audio packets are heavily based on the [QOA](https://qoaformat.org/) audio format, due to its ease of implementation, decent audio quality vs compression ratio, and high decoding performance.