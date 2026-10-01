/// SIFT 关键点。
///
/// `scale` 存放关键点的实际尺度（sigma，原始图像坐标），`scale_idx` 存放该关键点
/// 所在 octave 高斯金字塔中的层索引，用于描述符计算时定位梯度幅值与方向层。
///
/// 描述符不在关键点内，由 [`crate::Sift::detect_and_compute`] 单独返回。
#[derive(Debug, Clone)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub octave: usize,
    pub first_octave: i32,
    pub orientation: f32,
    pub scale_idx: usize,
}

impl KeyPoint {
    pub fn new(
        x: f32,
        y: f32,
        scale: f32,
        octave: usize,
        first_octave: i32,
        orientation: f32,
        scale_idx: usize,
    ) -> Self {
        KeyPoint {
            x,
            y,
            scale,
            octave,
            first_octave,
            orientation,
            scale_idx,
        }
    }
}
