pub struct VideoPlane {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>
}

impl VideoPlane {
    pub fn new(width: usize, height: usize) -> VideoPlane {
        VideoPlane { width: width, height: height, pixels: vec![0;width * height] }
    }

    pub fn from_slice(width: usize, height: usize, buffer: &[u8]) -> VideoPlane {
        assert!(buffer.len() == (width * height));
        let mut slice = VideoPlane::new(width, height);
        slice.pixels.copy_from_slice(buffer);

        slice
    }

    pub fn blit(self: &mut VideoPlane, src: &VideoPlane, dx: usize, dy: usize, sx: usize, sy: usize, sw: usize, sh: usize) {
        for row in 0..sh {
            let src_row = row + sy;
            let dest_row = row + dy;
            let src_offset = (src_row * src.width) + sx;
            let dst_offset = (dest_row * self.width) + dx;

            self.pixels[dst_offset..(dst_offset + sw)].copy_from_slice(&src.pixels[src_offset..(src_offset + sw)]);
        }
    }

    pub fn get_slice(self: &VideoPlane, sx: usize, sy: usize, sw: usize, sh: usize) -> VideoPlane {
        let mut new_slice = VideoPlane::new(sw, sh);
        new_slice.blit(self, 0, 0, sx, sy, sw, sh);

        new_slice
    }
}