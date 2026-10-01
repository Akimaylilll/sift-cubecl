use crate::descriptor::{self, DescriptorConfig};
use crate::detector::{self, SiftParams};
use crate::keypoint::KeyPoint;
use cubecl::prelude::*;
use image::DynamicImage;

/// SIFT 特征检测与描述（cubecl 实现）。
///
/// 同时提供 [`detect`](Self::detect) 与 [`detect_and_compute`](Self::detect_and_compute)
/// 两个方法，分别用于只提取关键点，或同时提取关键点与描述符。
///
/// 检测逻辑位于 [`detector`] 模块，描述符逻辑位于 [`descriptor`] 模块。
pub struct Sift<R: Runtime> {
    client: ComputeClient<R>,
    sigma: f32,
    descriptor_config: DescriptorConfig,
}

impl<R: Runtime> Sift<R> {
    /// 创建 SIFT 计算器。
    ///
    /// # Arguments
    ///
    /// * `device` - 计算设备（如 `cubecl::wgpu::WgpuDevice`）
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sift_cubecl::Sift;
    ///
    /// let device = cubecl::wgpu::WgpuDevice::default();
    /// let sift = Sift::new(device);
    /// ```
    pub fn new(device: R::Device) -> Self {
        let client = R::client(&device);
        Sift {
            client,
            sigma: 1.6,
            descriptor_config: DescriptorConfig::default(),
        }
    }

    /// 检测图像中的 SIFT 关键点。
    ///
    /// # Arguments
    ///
    /// * `image` - 输入灰度图像
    /// * `params` - SIFT 检测参数
    ///
    /// # Returns
    ///
    /// 检测到的关键点列表。
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sift_cubecl::{Sift, SiftParams};
    ///
    /// let sift = Sift::default();
    /// let params = SiftParams::default();
    /// // let image = image::open("lena.jpg").unwrap().grayscale();
    /// // let keypoints = sift.detect(image, &params);
    /// ```
    pub fn detect(&self, image: DynamicImage, params: &SiftParams) -> Vec<KeyPoint> {
        let (keypoints, _) = detector::detect(&self.client, self.sigma, image, params);
        keypoints
    }

    /// 检测关键点并计算描述符。
    ///
    /// # Arguments
    ///
    /// * `image` - 输入灰度图像
    /// * `params` - SIFT 检测参数（含描述符归一化方式 `normalization`）
    ///
    /// # Returns
    ///
    /// `(关键点列表, 描述符列表)`，两者按索引一一对应，描述符为 128 维向量。
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use sift_cubecl::{Sift, SiftParams};
    ///
    /// let sift = Sift::default();
    /// let params = SiftParams::default();
    /// // let image = image::open("lena.jpg").unwrap().grayscale();
    /// // let (keypoints, descriptors) = sift.detect_and_compute(image, &params);
    /// ```
    pub fn detect_and_compute(
        &self,
        image: DynamicImage,
        params: &SiftParams,
    ) -> (Vec<KeyPoint>, Vec<Vec<f32>>) {
        let (keypoints, scale_spaces) = detector::detect(&self.client, self.sigma, image, params);
        descriptor::compute_descriptors(
            &self.client,
            &self.descriptor_config,
            keypoints,
            &scale_spaces,
            params.normalization,
        )
    }
}

impl<R: Runtime> Default for Sift<R>
where
    R::Device: Default,
{
    fn default() -> Self {
        Self::new(R::Device::default())
    }
}
