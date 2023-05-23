use crate::plane::VideoPlane;

pub struct VideoFrame {
    pub width: usize,
    pub height: usize,
    pub plane_y: VideoPlane,
    pub plane_u: VideoPlane,
    pub plane_v: VideoPlane,
}

impl VideoFrame {
    pub fn new(width: usize, height: usize) -> VideoFrame {
        assert!(width % 2 == 0 && height % 2 == 0);

        let plane_y = VideoPlane::new(width, height);
        let mut plane_u = VideoPlane::new(width / 2, height / 2);
        let mut plane_v = VideoPlane::new(width / 2, height / 2);

        plane_u.pixels.fill(128);
        plane_v.pixels.fill(128);

        VideoFrame { width: width, height: height,
            plane_y: plane_y,
            plane_u: plane_u,
            plane_v: plane_v }
    }

    pub fn new_padded(width: usize, height: usize) -> VideoFrame {
        let pad_width: usize = width + (16 - (width % 16)) % 16;
        let pad_height = height + (16 - (height % 16)) % 16;

        let chroma_width = width / 2;
        let chroma_height = height / 2;

        let chroma_pad_width: usize = chroma_width + (16 - (chroma_width % 16)) % 16;
        let chroma_pad_height = chroma_height + (16 - (chroma_height % 16)) % 16;

        let plane_y = VideoPlane::new(pad_width, pad_height);
        let mut plane_u = VideoPlane::new(chroma_pad_width, chroma_pad_height);
        let mut plane_v = VideoPlane::new(chroma_pad_width, chroma_pad_height);

        plane_u.pixels.fill(128);
        plane_v.pixels.fill(128);

        VideoFrame { width: width, height: height,
            plane_y: plane_y,
            plane_u: plane_u,
            plane_v: plane_v }
    }

    pub fn from_planes(width: usize, height: usize, plane_y: VideoPlane, plane_u: VideoPlane, plane_v: VideoPlane) -> VideoFrame {
        assert!(plane_y.width == width && plane_y.height == height);
        assert!(plane_u.width == width && plane_u.height == height);
        assert!(plane_v.width == width && plane_v.height == height);

        VideoFrame { width: width, height: height,
            plane_y: plane_y,
            plane_u: plane_u.reduce(),
            plane_v: plane_v.reduce() }
    }
}