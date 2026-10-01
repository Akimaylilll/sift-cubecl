//! SIFT (Scale-Invariant Feature Transform) implementation in Rust, accelerated with
//! [cubecl](https://github.com/tracel-ai/cubecl).
//!
//! This library provides functionality to detect and describe keypoints in images
//! using the SIFT algorithm. The computation runs on any backend supported by cubecl.
//! Enable the corresponding feature to select the default backend (`wgpu` is enabled
//! by default):
//!
//! * `wgpu` — Vulkan / Metal / DX12 (default)
//! * `cuda` — NVIDIA CUDA
//! * `cpu` — cubecl's CPU runtime
//!
//! The generic type [`sift::Sift`] works with any [`cubecl::Runtime`]; [`Sift`] is a
//! type alias pinned to the backend selected by the enabled feature.
//!
//! # Examples
//!
//! ```rust,ignore
//! use sift_cubecl::{Sift, SiftParams};
//!
//! let sift = Sift::default();
//! let params = SiftParams { first_octave: -1, ..SiftParams::default() };
//!
//! let (keypoints, descriptors) = sift.detect_and_compute(image, &params);
//! ```

pub mod descriptor;
pub mod detector;
pub mod keypoint;
pub mod sift;

pub use descriptor::Normalization;
pub use detector::SiftParams;
pub use keypoint::KeyPoint;

/// 默认后端对应的 SIFT 类型（根据启用的 feature 自动选择）。
#[cfg(feature = "wgpu")]
pub type Sift = sift::Sift<cubecl::wgpu::WgpuRuntime>;

/// 默认后端对应的计算设备类型。
#[cfg(feature = "wgpu")]
pub type DefaultDevice = cubecl::wgpu::WgpuDevice;

#[cfg(all(not(feature = "wgpu"), feature = "cuda"))]
pub type Sift = sift::Sift<cubecl::cuda::CudaRuntime>;

#[cfg(all(not(feature = "wgpu"), feature = "cuda"))]
pub type DefaultDevice = cubecl::cuda::CudaDevice;

#[cfg(all(not(feature = "wgpu"), not(feature = "cuda"), feature = "cpu"))]
pub type Sift = sift::Sift<cubecl::cpu::CpuRuntime>;

#[cfg(all(not(feature = "wgpu"), not(feature = "cuda"), feature = "cpu"))]
pub type DefaultDevice = cubecl::cpu::CpuDevice;
