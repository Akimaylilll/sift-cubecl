use crate::descriptor::Normalization;
use crate::keypoint::KeyPoint;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use image::DynamicImage;
use nalgebra::{Matrix3, Vector3, LU};
use ndarray::{Array2, Array3, ArrayView2, Axis};

type LayerView<'a> = ArrayView2<'a, f32>;

struct ExtremaContext<'a> {
    prev: LayerView<'a>,
    current: LayerView<'a>,
    next: LayerView<'a>,
}

/// SIFT 检测参数。
#[derive(Debug, Clone)]
pub struct SiftParams {
    pub max_num_features: usize,
    pub first_octave: i32,
    pub num_octaves: usize,
    pub octave_resolution: usize,
    pub peak_threshold: f32,
    pub edge_threshold: f32,
    pub normalization: Normalization,
}

impl Default for SiftParams {
    fn default() -> Self {
        SiftParams {
            max_num_features: 0,
            first_octave: 0,
            num_octaves: 0,
            octave_resolution: 3,
            peak_threshold: 0.04,
            edge_threshold: 10.0,
            normalization: Normalization::L1,
        }
    }
}

/// 水平方向高斯模糊（GPU 内核）。
///
/// 每个线程处理一个像素，沿 x 方向卷积并做边缘镜像。
///
/// # Arguments
///
/// * `input` - 输入展平图像
/// * `output` - 输出展平图像
/// * `kernel` - 一维高斯核
/// * `width` / `height` - 图像尺寸
/// * `radius` - 高斯核半径
#[cube(launch)]
fn blur_horizontal(
    input: &Array<f32>,
    output: &mut Array<f32>,
    kernel: &Array<f32>,
    width: usize,
    height: usize,
    radius: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < width * height {
        let x = idx % width;
        let y = idx / width;

        let mut sum = 0.0;
        let radius_i = i32::cast_from(radius);
        let mut k = -radius_i;
        while k <= radius_i {
            let sample_x = i32::cast_from(x) + k;
            let mut sx = sample_x;
            if sample_x < 0 {
                sx = -sample_x;
            } else if sample_x >= i32::cast_from(width) {
                sx = 2 * i32::cast_from(width) - sample_x - 2;
            }
            if sx < 0 {
                sx = 0;
            }
            if sx >= i32::cast_from(width) {
                sx = i32::cast_from(width) - 1;
            }
            let k_idx = usize::cast_from(k + radius_i);
            sum += input[y * width + usize::cast_from(sx)] * kernel[k_idx];
            k += 1;
        }
        output[idx] = sum;
    }
}

/// 垂直方向高斯模糊（GPU 内核）。
///
/// 每个线程处理一个像素，沿 y 方向卷积并做边缘镜像。结果可写入输出缓冲区的
/// 指定偏移处（`offset + idx`），从而把各层拼进同一个展平金字塔，避免非对齐
/// 的子句柄绑定。
///
/// # Arguments
///
/// * `input` - 输入展平图像
/// * `output` - 输出展平缓冲区
/// * `kernel` - 一维高斯核
/// * `width` / `height` - 图像尺寸
/// * `radius` - 高斯核半径
/// * `offset` - 输出写入的起始偏移（元素数）
#[cube(launch)]
fn blur_vertical(
    input: &Array<f32>,
    output: &mut Array<f32>,
    kernel: &Array<f32>,
    width: usize,
    height: usize,
    radius: usize,
    offset: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < width * height {
        let x = idx % width;
        let y = idx / width;

        let mut sum = 0.0;
        let radius_i = i32::cast_from(radius);
        let mut k = -radius_i;
        while k <= radius_i {
            let sample_y = i32::cast_from(y) + k;
            let mut sy = sample_y;
            if sample_y < 0 {
                sy = -sample_y;
            } else if sample_y >= i32::cast_from(height) {
                sy = 2 * i32::cast_from(height) - sample_y - 2;
            }
            if sy < 0 {
                sy = 0;
            }
            if sy >= i32::cast_from(height) {
                sy = i32::cast_from(height) - 1;
            }
            let k_idx = usize::cast_from(k + radius_i);
            sum += input[usize::cast_from(sy) * width + x] * kernel[k_idx];
            k += 1;
        }
        output[offset + idx] = sum;
    }
}

/// 由高斯金字塔计算 DoG 金字塔（GPU 内核，逐元素相减）。
///
/// 每个线程处理一个 DoG 元素，`dog[s] = gaussian[s+1] - gaussian[s]`。
///
/// # Arguments
///
/// * `gaussian` - 展平的高斯金字塔（长度 `n_g * stride`）
/// * `dog` - 输出的展平 DoG 金字塔（长度 `n_dog * stride`）
/// * `scale_stride` - 单层像素数（`height * width`）
/// * `n_dog` - DoG 层数
#[cube(launch)]
fn compute_dog(gaussian: &Array<f32>, dog: &mut Array<f32>, scale_stride: usize, n_dog: usize) {
    let idx = ABSOLUTE_POS;
    if idx < n_dog * scale_stride {
        let s = idx / scale_stride;
        let off = idx % scale_stride;
        dog[idx] = gaussian[(s + 1) * scale_stride + off] - gaussian[s * scale_stride + off];
    }
}

/// 检测 DoG 金字塔中的极值点（GPU 内核）。
///
/// 每个线程处理一个 DoG 元素，判断其在 3×3×3 邻域内是否为局部极值，结果写入
/// `flags`（1 表示极值点）。使用标记数组而非原子压缩，以兼容所有后端
/// （`cubecl-cpu` 尚未实现 `Atomic` 类型）。
///
/// # Arguments
///
/// * `dog` - 展平的 DoG 金字塔（长度 `n_dog * stride`）
/// * `flags` - 输出标记（长度 `n_dog * stride`，1 表示极值点）
/// * `width` / `height` - 图像尺寸
/// * `num_dog_scales` - DoG 层数
/// * `scale_stride` - 单层像素数（`height * width`）
/// * `threshold` - 峰值阈值
#[cube(launch)]
fn detect_extrema(
    dog: &Array<f32>,
    flags: &mut Array<u32>,
    width: usize,
    height: usize,
    num_dog_scales: usize,
    scale_stride: usize,
    threshold: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_dog_scales * scale_stride {
        flags[idx] = 0;
        let s = idx / scale_stride;
        if s >= 1 && s < num_dog_scales - 1 {
            let rem = idx % scale_stride;
            let x = rem % width;
            let y = rem / width;
            if x >= 1 && y >= 1 && x < width - 1 && y < height - 1 {
                let center = dog[idx];

                let mut ge_count = f32::new(0.0_f32);
                let mut le_count = f32::new(0.0_f32);

                let mut dz = i32::new(0);
                while dz < 3 {
                    let mut dy = i32::new(0);
                    while dy < 3 {
                        let mut dx = i32::new(0);
                        while dx < 3 {
                            if dx == 1 && dy == 1 && dz == 1 {
                                ge_count += f32::new(0.0_f32);
                            } else {
                                let ns = i32::cast_from(s) + dz - 1;
                                let ny = i32::cast_from(y) + dy - 1;
                                let nx = i32::cast_from(x) + dx - 1;
                                let nidx = usize::cast_from(ns) * scale_stride
                                    + usize::cast_from(ny) * width
                                    + usize::cast_from(nx);
                                let v = dog[nidx];
                                if v >= center {
                                    ge_count += f32::new(1.0_f32);
                                }
                                if v <= center {
                                    le_count += f32::new(1.0_f32);
                                }
                            }
                            dx += 1;
                        }
                        dy += 1;
                    }
                    dz += 1;
                }

                if ge_count == f32::new(0.0_f32) && center >= threshold {
                    flags[idx] = 1;
                }
                if le_count == f32::new(0.0_f32) && center <= -threshold {
                    flags[idx] = 1;
                }
            }
        }
    }
}

/// 生成一维高斯核（归一化）。
///
/// # Arguments
///
/// * `sigma` - 高斯标准差
///
/// # Returns
///
/// `(半径, 归一化后的一维高斯核)`。
fn gaussian_kernel(sigma: f64) -> (usize, Vec<f32>) {
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;
    let sigma_sq_2 = 2.0 * sigma * sigma;

    let mut sum = 0.0f64;
    let mut kernel = Vec::with_capacity(kernel_size);
    for i in 0..kernel_size {
        let x = i as f64 - radius as f64;
        let weight = (-x * x / sigma_sq_2).exp();
        kernel.push(weight as f32);
        sum += weight;
    }
    for weight in &mut kernel {
        *weight = (*weight as f64 / sum) as f32;
    }

    (radius, kernel)
}

/// 单个 octave 的高斯金字塔（展平后保留在 GPU 显存中）。
///
/// 供描述符计算阶段直接消费，避免尺度空间在 CPU/GPU 之间来回搬运。
pub(crate) struct OctavePyramid {
    /// 展平的高斯金字塔，长度 `num_scales * height * width`（f32）。
    pub gaussian: cubecl::server::Handle,
    pub height: usize,
    pub width: usize,
    /// 高斯层数（`octave_resolution + 3`）。
    pub num_scales: usize,
}

/// 检测图像中的 SIFT 关键点，并返回每个 octave 的高斯金字塔（保留在 GPU 显存中）。
///
/// # Arguments
///
/// * `client` - cubecl 计算客户端
/// * `sigma` - 基础高斯尺度
/// * `image` - 输入灰度图像
/// * `params` - SIFT 检测参数
///
/// # Returns
///
/// `(关键点列表, 每个 octave 的高斯金字塔句柄)`。高斯金字塔留在显存中，
/// 供后续描述符计算直接使用，避免回读后再上传。
pub(crate) fn detect<R: Runtime>(
    client: &ComputeClient<R>,
    sigma: f32,
    image: DynamicImage,
    params: &SiftParams,
) -> (Vec<KeyPoint>, Vec<OctavePyramid>) {
    let mut image = dynamic_image_to_ndarray(&image);

    if params.first_octave < 0 {
        let (h, w) = image.dim();
        image = resize_bilinear_grayscale(&image, h * 2, w * 2);
    }

    let mut dog_pyramids: Vec<Array3<f32>> = Vec::new();
    let mut octave_pyramids: Vec<OctavePyramid> = Vec::new();
    let (image_width, image_height) = (image.shape()[1], image.shape()[0]);

    let num_octaves = if params.num_octaves > 0 {
        params.num_octaves
    } else {
        let mut image_size = image_width.min(image_height);
        let mut n = 0;
        while image_size >= 8 {
            n += 1;
            image_size /= 2;
        }
        n
    };
    let num_scales = params.octave_resolution;
    let n_dog = params.octave_resolution + 2;
    let n_g = params.octave_resolution + 3;

    let mut candidates: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut current_img = image.clone();

    for octave in 0..num_octaves {
        let (w, h) = if octave == 0 {
            (image_width, image_height)
        } else {
            (current_img.shape()[1], current_img.shape()[0])
        };
        let stride = h * w;

        // GPU 全链路：高斯金字塔 → DoG → 极值检测（输出标记数组）
        let gaussian = compute_gaussian_pyramid(client, sigma, &current_img, n_g, h, w, num_scales);
        let flags = find_extrema(client, &gaussian, w, h, n_dog, params.peak_threshold);

        // 批量读回：完整高斯金字塔 + 极值标记（整块对齐，避免非对齐子句柄）
        let bytes_vec = client.read(vec![gaussian.clone(), flags]);

        // 高斯金字塔 → Array3（供 CPU 计算 DoG 精化 + 降采样）
        let gaussian_flat = f32::from_bytes(&bytes_vec[0]);
        let gaussian_pyramid = Array3::from_shape_vec((n_g, h, w), gaussian_flat.to_vec()).unwrap();
        dog_pyramids.push(compute_dog_pyramid(&gaussian_pyramid, n_dog));

        // 降采样
        if octave < num_octaves - 1 {
            let downsample_source_idx = num_scales;
            let layer = gaussian_pyramid
                .index_axis(Axis(0), downsample_source_idx)
                .to_owned();
            current_img = resize_nearest_grayscale_simple(&layer, h / 2, w / 2);
        }

        // 从标记数组展开候选点
        let flags_data = u32::from_bytes(&bytes_vec[1]);
        for (i, &f) in flags_data.iter().enumerate() {
            if f == 1 {
                let s = i / stride;
                let rem = i % stride;
                let x = rem % w;
                let y = rem / w;
                candidates.push((octave, s, x, y));
            }
        }

        octave_pyramids.push(OctavePyramid {
            gaussian,
            height: h,
            width: w,
            num_scales: n_g,
        });
    }

    let keypoints = refine_candidates(&dog_pyramids, &candidates, params, sigma, num_scales);

    (keypoints, octave_pyramids)
}

/// 在 GPU 上计算单 octave 的高斯金字塔，结果保留在显存中。
///
/// 每一层独立地对输入图像做不同 σ 的高斯模糊，所有层写入同一个展平缓冲区
/// （长度 `n_g * stride`），返回单个 GPU 句柄。数据不读回 CPU，供后续 DoG、
/// 极值检测与描述符计算直接在 GPU 上链式使用。
///
/// # Arguments
///
/// * `client` - cubecl 计算客户端
/// * `sigma` - 基础高斯尺度
/// * `input` - 当前 octave 的输入图像
/// * `n_g` - 高斯层数（`octave_resolution + 3`）
/// * `h` / `w` - 图像尺寸
/// * `num_scales` - 每个 octave 的尺度数（`octave_resolution`）
///
/// # Returns
///
/// 展平的高斯金字塔句柄（长度 `n_g * h * w`）。
fn compute_gaussian_pyramid<R: Runtime>(
    client: &ComputeClient<R>,
    sigma: f32,
    input: &Array2<f32>,
    n_g: usize,
    h: usize,
    w: usize,
    num_scales: usize,
) -> cubecl::server::Handle {
    let stride = h * w;
    let elem = core::mem::size_of::<f32>();

    let flat: Vec<f32> = input.iter().copied().collect();
    let input_handle = client.create_from_slice(f32::as_bytes(&flat));
    let temp = client.empty(stride * elem);

    let gaussian = client.empty(n_g * stride * elem);

    for scale in 0..n_g {
        let cube_dim = CubeDim::new(client, stride);
        let cube_count = calculate_cube_count_elemwise(client, stride, cube_dim);

        let k = 2.0_f32.powf(scale as f32 / num_scales as f32);
        let blurred_sigma = sigma * k;
        let (radius, kernel) = gaussian_kernel(blurred_sigma as f64);
        let kernel_handle = client.create_from_slice(f32::as_bytes(&kernel));

        unsafe {
            blur_horizontal::launch::<R>(
                client,
                cube_count.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(input_handle.clone(), stride),
                ArrayArg::from_raw_parts(temp.clone(), stride),
                ArrayArg::from_raw_parts(kernel_handle.clone(), kernel.len()),
                w,
                h,
                radius,
            );
            blur_vertical::launch::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(temp.clone(), stride),
                ArrayArg::from_raw_parts(gaussian.clone(), n_g * stride),
                ArrayArg::from_raw_parts(kernel_handle, kernel.len()),
                w,
                h,
                radius,
                scale * stride,
            );
        }
    }

    gaussian
}

/// 在 GPU 上由高斯金字塔计算 DoG 并检测极值点，输出标记数组。
///
/// 全程数据不离开显存：先对展平高斯金字塔做 DoG（`compute_dog`），再对 DoG
/// 做极值检测（`detect_extrema`），结果写入 `n_dog * stride` 个 `u32` 的标记
/// 数组（1 表示极值点）。
///
/// # Arguments
///
/// * `client` - cubecl 计算客户端
/// * `gaussian` - 展平高斯金字塔句柄（长度 `n_g * stride`）
/// * `w` / `h` - 图像尺寸
/// * `n_dog` - DoG 层数
/// * `threshold` - 峰值阈值
///
/// # Returns
///
/// 标记数组的 GPU 句柄（长度 `n_dog * stride`）。
fn find_extrema<R: Runtime>(
    client: &ComputeClient<R>,
    gaussian: &cubecl::server::Handle,
    w: usize,
    h: usize,
    n_dog: usize,
    threshold: f32,
) -> cubecl::server::Handle {
    let stride = h * w;
    let elem = core::mem::size_of::<f32>();

    // 阶段一：DoG（`dog[s] = gaussian[s+1] - gaussian[s]`）
    let dog = client.empty(n_dog * stride * elem);
    let cube_dim = CubeDim::new(client, n_dog * stride);
    let cube_count = calculate_cube_count_elemwise(client, n_dog * stride, cube_dim);
    unsafe {
        compute_dog::launch::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts(gaussian.clone(), (n_dog + 1) * stride),
            ArrayArg::from_raw_parts(dog.clone(), n_dog * stride),
            stride,
            n_dog,
        );
    }

    // 阶段二：极值检测，写入标记数组
    let flags = client.empty(n_dog * stride * core::mem::size_of::<u32>());

    let total = n_dog * stride;
    let cube_dim = CubeDim::new(client, total);
    let cube_count = calculate_cube_count_elemwise(client, total, cube_dim);
    unsafe {
        detect_extrema::launch::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(dog.clone(), total),
            ArrayArg::from_raw_parts(flags.clone(), total),
            w,
            h,
            n_dog,
            stride,
            threshold,
        );
    }

    flags
}

/// 由高斯金字塔计算 DoG 金字塔（`dog[scale] = gaussian[scale+1] - gaussian[scale]`）。
///
/// 仅用于 CPU 侧的亚像素精化，利用已读回的高斯金字塔逐元素相减（零传输）。
///
/// # Arguments
///
/// * `gaussian` - 高斯金字塔，形状为 `[scales, height, width]`
/// * `n_dog` - DoG 层数（通常为 `octave_resolution + 2`）
///
/// # Returns
///
/// DoG 金字塔，形状为 `[n_dog, height, width]`。
fn compute_dog_pyramid(gaussian: &Array3<f32>, n_dog: usize) -> Array3<f32> {
    let (_, h, w) = gaussian.dim();
    let mut dog_pyramid = Array3::<f32>::zeros((n_dog, h, w));
    for scale in 1..n_dog + 1 {
        let a_channel = gaussian.index_axis(Axis(0), scale);
        let b_channel = gaussian.index_axis(Axis(0), scale - 1);
        let mut result_channel = dog_pyramid.index_axis_mut(Axis(0), scale - 1);
        result_channel.assign(&(&a_channel - &b_channel));
    }
    dog_pyramid
}

/// 将灰度 `DynamicImage` 转换为 `f32` 二维数组。
///
/// # Arguments
///
/// * `img` - 输入图像（须为 `ImageLuma8`）
///
/// # Returns
///
/// 形状为 `[height, width]` 的灰度二维数组。
///
/// # Panics
///
/// 传入非 `ImageLuma8` 格式时 panic。
fn dynamic_image_to_ndarray(img: &DynamicImage) -> Array2<f32> {
    match img {
        DynamicImage::ImageLuma8(buffer) => {
            let (width, height) = buffer.dimensions();
            Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
                buffer.get_pixel(x as u32, y as u32)[0] as f32
            })
        }
        _ => panic!("Unsupported image format"),
    }
}

/// 最近邻插值缩放灰度图。
///
/// # Arguments
///
/// * `image` - 输入灰度图
/// * `new_height` - 目标高度
/// * `new_width` - 目标宽度
///
/// # Returns
///
/// 缩放后的灰度图。
fn resize_nearest_grayscale_simple(
    image: &Array2<f32>,
    new_height: usize,
    new_width: usize,
) -> Array2<f32> {
    let (src_h, src_w) = image.dim();
    let scale_y = src_h as f32 / new_height as f32;
    let scale_x = src_w as f32 / new_width as f32;

    Array2::from_shape_fn((new_height, new_width), |(y, x)| {
        let src_y = (y as f32 * scale_y) as usize;
        let src_x = (x as f32 * scale_x) as usize;
        image[[src_y.min(src_h - 1), src_x.min(src_w - 1)]]
    })
}

/// 双线性插值缩放灰度图（用于上采样）。
///
/// # Arguments
///
/// * `image` - 输入灰度图
/// * `new_height` - 目标高度
/// * `new_width` - 目标宽度
///
/// # Returns
///
/// 缩放后的灰度图。
fn resize_bilinear_grayscale(
    image: &Array2<f32>,
    new_height: usize,
    new_width: usize,
) -> Array2<f32> {
    let (src_h, src_w) = image.dim();
    let scale_y = src_h as f32 / new_height as f32;
    let scale_x = src_w as f32 / new_width as f32;

    Array2::from_shape_fn((new_height, new_width), |(y, x)| {
        let src_y = (y as f32 + 0.5) * scale_y - 0.5;
        let src_x = (x as f32 + 0.5) * scale_x - 0.5;

        let x0 = src_x.floor().max(0.0) as usize;
        let y0 = src_y.floor().max(0.0) as usize;
        let x1 = (x0 + 1).min(src_w - 1);
        let y1 = (y0 + 1).min(src_h - 1);

        let fx = src_x - src_x.floor();
        let fy = src_y - src_y.floor();

        let v00 = image[[y0, x0]];
        let v01 = image[[y0, x1]];
        let v10 = image[[y1, x0]];
        let v11 = image[[y1, x1]];

        v00 * (1.0 - fx) * (1.0 - fy)
            + v01 * fx * (1.0 - fy)
            + v10 * (1.0 - fx) * fy
            + v11 * fx * fy
    })
}

/// 对候选极值点做亚像素精化（牛顿迭代求解精确位置）。
///
/// 通过二阶泰勒展开与 Hessian 矩阵迭代求解极值点的亚像素偏移量。
///
/// # Arguments
///
/// * `ctx` - 当前点所在的前/中/后三层 DoG 视图
/// * `x` / `y` - 候选点整数坐标
/// * `scale` - 候选点所在尺度索引
/// * `dog_scales` - DoG 总层数
///
/// # Returns
///
/// 收敛时返回 `(dx, dy, ds, dog_value)`，否则返回 `None`。
fn refine_extremum(
    ctx: &ExtremaContext,
    x: usize,
    y: usize,
    scale: usize,
    dog_scales: usize,
) -> Option<(f32, f32, f32, f32)> {
    const MAX_ITER: usize = 5;
    const CONVERGE_THRESH: f32 = 0.5;

    let mut offset = Vector3::new(0., 0., 0.);
    let (width, height) = (ctx.current.shape()[1], ctx.current.shape()[0]);

    let mut x_float = x as f32;
    let mut y_float = y as f32;
    let mut s_float = scale as f32;

    for _iter in 0..MAX_ITER {
        let x_int = x_float as usize;
        let y_int = y_float as usize;
        let s_int = s_float as usize;

        // 检查边界 - DOG金字塔的边界检查
        if x_int < 1
            || x_int >= width - 1
            || y_int < 1
            || y_int >= height - 1
            || s_int < 1
            || s_int >= dog_scales - 1
        {
            return None;
        }

        // 计算一阶导数（梯度）
        let dx = (ctx.current[[y_int, x_int + 1]] - ctx.current[[y_int, x_int - 1]]) * 0.5;
        let dy = (ctx.current[[y_int + 1, x_int]] - ctx.current[[y_int - 1, x_int]]) * 0.5;
        let ds = (ctx.next[[y_int, x_int]] - ctx.prev[[y_int, x_int]]) * 0.5;

        // 计算二阶导数（Hessian矩阵）
        let dxx = ctx.current[[y_int, x_int + 1]] - 2.0 * ctx.current[[y_int, x_int]]
            + ctx.current[[y_int, x_int - 1]];
        let dyy = ctx.current[[y_int + 1, x_int]] - 2.0 * ctx.current[[y_int, x_int]]
            + ctx.current[[y_int - 1, x_int]];
        let dss =
            ctx.next[[y_int, x_int]] - 2.0 * ctx.current[[y_int, x_int]] + ctx.prev[[y_int, x_int]];

        let dxy = (ctx.current[[y_int + 1, x_int + 1]]
            - ctx.current[[y_int + 1, x_int - 1]]
            - ctx.current[[y_int - 1, x_int + 1]]
            + ctx.current[[y_int - 1, x_int - 1]])
            * 0.25;
        let dxs = (ctx.next[[y_int, x_int + 1]]
            - ctx.next[[y_int, x_int - 1]]
            - ctx.prev[[y_int, x_int + 1]]
            + ctx.prev[[y_int, x_int - 1]])
            * 0.25;
        let dys = (ctx.next[[y_int + 1, x_int]]
            - ctx.next[[y_int - 1, x_int]]
            - ctx.prev[[y_int + 1, x_int]]
            + ctx.prev[[y_int - 1, x_int]])
            * 0.25;

        let hessian = Matrix3::new(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);

        let lu = LU::new(hessian);
        if lu.determinant().abs() < 1e-6 {
            return None; // Hessian不可逆
        }

        let grad = Vector3::new(dx, dy, ds);
        let delta = lu.solve(&(-grad)).unwrap();

        // 更新偏移量
        offset += delta;

        // 更新坐标
        x_float += delta[0];
        y_float += delta[1];
        s_float += delta[2];

        // 收敛判断
        if delta.norm() < CONVERGE_THRESH {
            // 检查最终位置是否在边界内
            if x_float < 0.0
                || x_float >= (width as f32)
                || y_float < 0.0
                || y_float >= (height as f32)
                || s_float < 0.0
                || s_float >= dog_scales as f32
            {
                return None;
            }

            // 计算在极值点的函数值（使用完整的二阶泰勒展开）
            let dog_value = ctx.current[[y_int, x_int]]
                + grad.dot(&offset)
                + 0.5 * (offset.transpose() * hessian * offset)[0];

            // 返回相对于当前点的偏移量
            return Some((delta[0], delta[1], delta[2], dog_value));
        }
    }
    None
}

/// 判断某点是否为边缘响应（依据 Hessian 矩阵迹与行列式之比）。
///
/// # Arguments
///
/// * `image` - DoG 层视图
/// * `x` / `y` - 检测点坐标
/// * `edge_threshold` - 边缘响应阈值
///
/// # Returns
///
/// 是否为边缘响应（应被剔除）。
fn is_edge_response(image: LayerView, x: usize, y: usize, edge_threshold: f32) -> bool {
    let dxx = image[[y, x + 1]] + image[[y, x - 1]] - 2.0 * image[[y, x]];
    let dyy = image[[y + 1, x]] + image[[y - 1, x]] - 2.0 * image[[y, x]];
    let dxy = (image[[y + 1, x + 1]] - image[[y + 1, x - 1]] - image[[y - 1, x + 1]]
        + image[[y - 1, x - 1]])
        / 4.0;

    let trace = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;

    if det <= 0.0 {
        return true;
    }

    (trace * trace) / det > (edge_threshold + 1.0).powi(2) / edge_threshold
}

/// 由候选极值点（整数坐标）精化并筛选关键点。
///
/// 对每个候选点做亚像素精化、峰值阈值过滤与边缘响应剔除，并按对比度
/// 排序、限制最大数量。
///
/// # Arguments
///
/// * `dog_pyramids` - 每个 octave 的 DoG 金字塔
/// * `candidates` - 候选点列表，每项为 `(octave, scale, x, y)`
/// * `params` - SIFT 检测参数
/// * `sigma` - 基础高斯尺度
/// * `num_scales` - 每个 octave 的尺度数
///
/// # Returns
///
/// 精化后的关键点列表。
fn refine_candidates(
    dog_pyramids: &[Array3<f32>],
    candidates: &[(usize, usize, usize, usize)],
    params: &SiftParams,
    sigma: f32,
    num_scales: usize,
) -> Vec<KeyPoint> {
    let dog_scales = num_scales + 2;
    let mut keypoints: Vec<(f32, KeyPoint)> = Vec::new();

    for &(octave, scale, x, y) in candidates {
        let dog = &dog_pyramids[octave];
        let ctx = ExtremaContext {
            prev: dog.index_axis(Axis(0), scale - 1),
            current: dog.index_axis(Axis(0), scale),
            next: dog.index_axis(Axis(0), scale + 1),
        };

        if let Some((dx, dy, _ds, value)) = refine_extremum(&ctx, x, y, scale, dog_scales) {
            // 检查偏移量是否过大（表示插值不可靠）
            if dx.abs() >= 1.0 || dy.abs() >= 1.0 || _ds.abs() >= 1.0 {
                continue;
            }
            if value.abs() < params.peak_threshold {
                continue;
            }

            // 消除边缘响应
            if is_edge_response(ctx.current, x, y, params.edge_threshold) {
                continue;
            }

            // 计算实际尺度值：sigma * 2^(first_octave + octave + (scale + ds)/octave_resolution)
            let scale_factor = 2.0_f32.powf(
                params.first_octave as f32
                    + octave as f32
                    + (scale as f32 + _ds) / params.octave_resolution as f32,
            );
            let actual_scale = sigma * scale_factor;

            // 坐标缩放因子：2^(first_octave + octave)
            let coord_scale = 2.0_f32.powi(params.first_octave + octave as i32);

            keypoints.push((
                value.abs(),
                KeyPoint {
                    x: (x as f32 + dx) * coord_scale,
                    y: (y as f32 + dy) * coord_scale,
                    scale: actual_scale,
                    octave,
                    first_octave: params.first_octave,
                    orientation: 0.0,
                    // DoG 尺度 `scale` 对应高斯层 `scale`（dog[s] = gaussian[s+1] - gaussian[s]）
                    scale_idx: scale,
                },
            ));
        }
    }

    // 按响应值（对比度）从大到小排序
    keypoints.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // 限制最大特征点数量，0表示不限制
    if params.max_num_features > 0 && keypoints.len() > params.max_num_features {
        keypoints.truncate(params.max_num_features);
    }

    keypoints.into_iter().map(|(_, kp)| kp).collect()
}
