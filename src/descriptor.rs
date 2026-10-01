#![allow(clippy::approx_constant)]

use crate::detector::OctavePyramid;
use crate::keypoint::KeyPoint;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

/// 描述符归一化方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// L1 范数归一化
    L1,
    /// L2 范数归一化（含截断与二次归一化）
    L2,
}

/// 描述符计算的配置参数。
pub(crate) struct DescriptorConfig {
    pub gaussian_sigma_factor: f32,
    pub num_bins: usize,
    pub peak_ratio: f32,
    pub descriptor_width: usize,
    pub descriptor_num_bins: usize,
}

impl Default for DescriptorConfig {
    fn default() -> Self {
        DescriptorConfig {
            gaussian_sigma_factor: 1.5,
            num_bins: 36,
            peak_ratio: 0.8,
            descriptor_width: 4,
            descriptor_num_bins: 8,
        }
    }
}

/// 每个八度的梯度缓冲
struct OctaveGrad<R: Runtime> {
    mag: cubecl::server::Handle,
    ori: cubecl::server::Handle,
    num_scales: usize,
    height: usize,
    width: usize,
    _marker: core::marker::PhantomData<R>,
}

/// 计算单个尺度层每个像素的梯度幅值与方向（GPU 内核）。
///
/// 每个线程处理一个像素，使用中心差分计算 dx、dy，得到梯度幅值与方向
/// （标准化到 `[0, 2π)`），结果写入对应的展平缓冲区。
///
/// # Arguments
///
/// * `scale_space` - 展平的尺度空间数据
/// * `magnitudes` - 输出的梯度幅值缓冲区
/// * `orientations` - 输出的梯度方向缓冲区
/// * `width` - 图像宽度
/// * `height` - 图像高度
#[cube(launch)]
fn compute_gradients(
    scale_space: &Array<f32>,
    magnitudes: &mut Array<f32>,
    orientations: &mut Array<f32>,
    width: usize,
    height: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < scale_space.len() {
        let x = idx % width;
        let y = (idx / width) % height;

        if x > 0 && y > 0 && x < width - 1 && y < height - 1 {
            let dx = scale_space[idx + 1] - scale_space[idx - 1];
            let dy = scale_space[idx + width] - scale_space[idx - width];

            magnitudes[idx] = (dx * dx + dy * dy).sqrt();

            let mut orientation = dy.atan2(dx);
            if orientation < 0.0 {
                orientation += 6.283_185_3;
            }
            orientations[idx] = orientation;
        } else {
            magnitudes[idx] = 0.0;
            orientations[idx] = 0.0;
        }
    }
}

/// 计算每个关键点的方向直方图（GPU 内核）。
///
/// 每个线程处理一个关键点，在其高斯加权窗口内采样，将加权梯度累积到
/// 对应的方向直方图中。
///
/// # Arguments
///
/// * `magnitudes` - 梯度幅值缓冲区
/// * `orientations` - 梯度方向缓冲区
/// * `kp_x` / `kp_y` - 关键点坐标
/// * `kp_scale` - 关键点尺度
/// * `kp_octave_scale` - 从原图坐标到金字塔坐标的缩放因子
/// * `kp_scale_idx` - 关键点所在尺度索引
/// * `histogram` - 输出的直方图缓冲区（`num_keypoints * num_bins`）
/// * `width` / `height` - 图像尺寸
/// * `scale_stride` - 单个尺度的像素数（`height * width`）
/// * `num_bins` - 方向 bin 数
/// * `gaussian_sigma_factor` - 高斯窗口尺度因子
#[cube(launch)]
fn compute_orientation_histogram(
    magnitudes: &Array<f32>,
    orientations: &Array<f32>,
    kp_x: &Array<f32>,
    kp_y: &Array<f32>,
    kp_scale: &Array<f32>,
    kp_octave_scale: &Array<f32>,
    kp_scale_idx: &Array<u32>,
    histogram: &mut Array<f32>,
    width: usize,
    height: usize,
    scale_stride: usize,
    num_bins: usize,
    gaussian_sigma_factor: f32,
) {
    let kp_idx = ABSOLUTE_POS;
    if kp_idx < kp_x.len() {
        let x = kp_x[kp_idx];
        let y = kp_y[kp_idx];
        let scale = kp_scale[kp_idx];
        let octave_scale = kp_octave_scale[kp_idx];
        let scale_idx = usize::cast_from(kp_scale_idx[kp_idx]);

        let sigma = gaussian_sigma_factor * scale;
        let radius = i32::cast_from((3.0 * sigma).round());

        let mut b = 0;
        while b < num_bins {
            histogram[kp_idx * num_bins + b] = 0.0;
            b += 1;
        }

        let base = scale_idx * scale_stride;
        let cx = x / octave_scale;
        let cy = y / octave_scale;

        let mut dy = -radius;
        while dy <= radius {
            let mut dx = -radius;
            while dx <= radius {
                let sample_x = i32::cast_from((cx + f32::cast_from(dx)).round());
                let sample_y = i32::cast_from((cy + f32::cast_from(dy)).round());

                if sample_x >= 0
                    && sample_x < i32::cast_from(width)
                    && sample_y >= 0
                    && sample_y < i32::cast_from(height)
                {
                    let sx = usize::cast_from(sample_x);
                    let sy = usize::cast_from(sample_y);
                    let idx = base + sy * width + sx;

                    let magnitude = magnitudes[idx];
                    let orientation = orientations[idx];

                    let weight = (-f32::cast_from(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                    let weighted = magnitude * weight;

                    let angle_bin = orientation * f32::cast_from(num_bins) / 6.283_185_3;
                    let bin = usize::cast_from(angle_bin) % num_bins;
                    let fraction = angle_bin - f32::cast_from(bin);

                    histogram[kp_idx * num_bins + bin] += weighted * (1.0 - fraction);
                    histogram[kp_idx * num_bins + (bin + 1) % num_bins] += weighted * fraction;
                }

                dx += 1;
            }
            dy += 1;
        }
    }
}

/// 计算每个关键点的 128 维描述符（GPU 内核）。
///
/// 每个线程处理一个关键点，在 16×16 邻域内采样并旋转到主方向，按 4×4 cell
/// 与 8 个方向 bin 做三线性插值累积。
///
/// # Arguments
///
/// * `magnitudes` - 梯度幅值缓冲区
/// * `orientations` - 梯度方向缓冲区
/// * `kp_x` / `kp_y` - 关键点坐标
/// * `kp_scale` - 关键点尺度
/// * `kp_octave_scale` - 从原图坐标到金字塔坐标的缩放因子
/// * `kp_scale_idx` - 关键点所在尺度索引
/// * `kp_orientation` - 关键点主方向
/// * `descriptors` - 输出的描述符缓冲区（`num_keypoints * desc_len`）
/// * `width` / `height` - 图像尺寸
/// * `scale_stride` - 单个尺度的像素数（`height * width`）
/// * `descriptor_width` - 描述符每边 cell 数（默认 4）
/// * `descriptor_num_bins` - 每 cell 的方向 bin 数（默认 8）
#[cube(launch)]
fn compute_descriptor(
    magnitudes: &Array<f32>,
    orientations: &Array<f32>,
    kp_x: &Array<f32>,
    kp_y: &Array<f32>,
    kp_scale: &Array<f32>,
    kp_octave_scale: &Array<f32>,
    kp_scale_idx: &Array<u32>,
    kp_orientation: &Array<f32>,
    descriptors: &mut Array<f32>,
    width: usize,
    height: usize,
    scale_stride: usize,
    descriptor_width: usize,
    descriptor_num_bins: usize,
) {
    let kp_idx = ABSOLUTE_POS;
    if kp_idx < kp_x.len() {
        let x = kp_x[kp_idx];
        let y = kp_y[kp_idx];
        let scale = kp_scale[kp_idx];
        let octave_scale = kp_octave_scale[kp_idx];
        let scale_idx = usize::cast_from(kp_scale_idx[kp_idx]);
        let orientation = kp_orientation[kp_idx];

        let descriptor_size = 16.0 * scale;
        let cell_size = descriptor_size / f32::cast_from(descriptor_width);
        let cos_theta = orientation.cos();
        let sin_theta = orientation.sin();
        let sigma = 0.5 * descriptor_size;
        let invsig2 = 1.0 / (2.0 * sigma * sigma);
        let half_size = i32::cast_from((descriptor_size / 2.0).ceil());

        let base = scale_idx * scale_stride;
        let width_f = f32::cast_from(width);
        let height_f = f32::cast_from(height);
        let descriptor_width_f = f32::cast_from(descriptor_width);
        let descriptor_num_bins_f = f32::cast_from(descriptor_num_bins);
        let descriptor_width_i32 = i32::cast_from(descriptor_width);
        let descriptor_num_bins_i32 = i32::cast_from(descriptor_num_bins);
        let desc_total = descriptor_width * descriptor_width * descriptor_num_bins;

        let mut d = 0;
        while d < desc_total {
            descriptors[kp_idx * desc_total + d] = 0.0;
            d += 1;
        }

        let mut dy = -half_size;
        while dy < half_size {
            let mut dx = -half_size;
            while dx < half_size {
                let rx = f32::cast_from(dx);
                let ry = f32::cast_from(dy);

                let tr_x = cos_theta * rx - sin_theta * ry;
                let tr_y = sin_theta * rx + cos_theta * ry;

                let sx = x / octave_scale + tr_x;
                let sy = y / octave_scale + tr_y;

                if sx >= 1.0 && sy >= 1.0 && sx < width_f - 1.0 && sy < height_f - 1.0 {
                    let win_weight = (-f32::cast_from(dx * dx + dy * dy) * invsig2).exp();

                    let xi = usize::cast_from(sx.floor());
                    let yi = usize::cast_from(sy.floor());

                    if xi < width - 1 && yi < height - 1 {
                        let xf = sx - f32::cast_from(xi);
                        let yf = sy - f32::cast_from(yi);

                        let idx00 = base + yi * width + xi;
                        let mag_interp = magnitudes[idx00] * (1.0 - xf) * (1.0 - yf)
                            + magnitudes[idx00 + 1] * xf * (1.0 - yf)
                            + magnitudes[idx00 + width] * (1.0 - xf) * yf
                            + magnitudes[idx00 + width + 1] * xf * yf;
                        let ori_interp = orientations[idx00] * (1.0 - xf) * (1.0 - yf)
                            + orientations[idx00 + 1] * xf * (1.0 - yf)
                            + orientations[idx00 + width] * (1.0 - xf) * yf
                            + orientations[idx00 + width + 1] * xf * yf;

                        let mut angle_diff = ori_interp - orientation;
                        if angle_diff < 0.0 {
                            angle_diff += 6.283_185_3;
                        }

                        let ori_bin = angle_diff * descriptor_num_bins_f / 6.283_185_3;

                        let cx = (tr_x + descriptor_size / 2.0) / cell_size;
                        let cy = (tr_y + descriptor_size / 2.0) / cell_size;

                        if cx >= -1.0
                            && cx < descriptor_width_f
                            && cy >= -1.0
                            && cy < descriptor_width_f
                        {
                            let c0 = i32::cast_from((cx - 0.5).max(-1.0).min(descriptor_width_f));
                            let r0 = i32::cast_from((cy - 0.5).max(-1.0).min(descriptor_width_f));
                            let o0 = i32::cast_from(ori_bin);

                            let dc = (cx - 0.5) - f32::cast_from(c0);
                            let dr = (cy - 0.5) - f32::cast_from(r0);
                            let do_ = ori_bin - f32::cast_from(o0);

                            let mut rr = r0;
                            while rr <= r0 + 1 {
                                if rr >= 0 && rr < descriptor_width_i32 {
                                    let mut wr = 1.0 - dr;
                                    if rr != r0 {
                                        wr = dr;
                                    }
                                    let mut cc = c0;
                                    while cc <= c0 + 1 {
                                        if cc >= 0 && cc < descriptor_width_i32 {
                                            let mut wc = 1.0 - dc;
                                            if cc != c0 {
                                                wc = dc;
                                            }
                                            let mut oo = o0;
                                            while oo <= o0 + 1 {
                                                let oo_wrap = (oo + descriptor_num_bins_i32)
                                                    % descriptor_num_bins_i32;
                                                let mut wo = 1.0 - do_;
                                                if oo != o0 {
                                                    wo = do_;
                                                }
                                                let didx = (usize::cast_from(rr)
                                                    * descriptor_width
                                                    + usize::cast_from(cc))
                                                    * descriptor_num_bins
                                                    + usize::cast_from(oo_wrap);
                                                descriptors[kp_idx * desc_total + didx] +=
                                                    mag_interp * win_weight * wr * wc * wo;
                                                oo += 1;
                                            }
                                        }
                                        cc += 1;
                                    }
                                }
                                rr += 1;
                            }
                        }
                    }
                }

                dx += 1;
            }
            dy += 1;
        }
    }
}

/// 对方向直方图做循环平滑，权重为 `[0.25, 0.5, 0.25]`。
///
/// # Arguments
///
/// * `histogram` - 原始方向直方图
///
/// # Returns
///
/// 平滑后的直方图。
fn smooth_histogram(histogram: &[f32]) -> Vec<f32> {
    let mut smoothed = vec![0.0; histogram.len()];
    for i in 0..histogram.len() {
        let prev = histogram[(i + histogram.len() - 1) % histogram.len()];
        let curr = histogram[i];
        let next = histogram[(i + 1) % histogram.len()];
        smoothed[i] = 0.25 * prev + 0.5 * curr + 0.25 * next;
    }
    smoothed
}

/// 从方向直方图中找出超过峰阈值的候选主方向。
///
/// 平滑直方图后取超过 `peak_ratio * max` 阈值的局部极大值，并通过抛物线
/// 插值精确定位角度。
///
/// # Arguments
///
/// * `histogram` - 方向直方图
/// * `num_bins` - 方向 bin 数
/// * `peak_ratio` - 峰值阈值比例
///
/// # Returns
///
/// 候选主方向角度列表（弧度，范围 `[0, 2π)`）。
fn find_peaks(histogram: &[f32], num_bins: usize, peak_ratio: f32) -> Vec<f32> {
    let smoothed = smooth_histogram(histogram);
    let max_value = smoothed.iter().cloned().fold(0.0, f32::max);
    let threshold = max_value * peak_ratio;

    let mut peaks = Vec::new();
    for i in 0..num_bins {
        let prev = smoothed[(i + num_bins - 1) % num_bins];
        let curr = smoothed[i];
        let next = smoothed[(i + 1) % num_bins];

        if curr > prev && curr > next && curr >= threshold {
            let denominator = prev - 2.0 * curr + next;
            if denominator.abs() > 1e-6 {
                let interpolated_bin = i as f32 + 0.5 * (prev - next) / denominator;
                let interpolated_bin = (interpolated_bin + num_bins as f32) % num_bins as f32;
                let angle_rad = interpolated_bin * 6.283_185_3 / num_bins as f32;
                peaks.push(angle_rad);
            }
        }
    }

    if peaks.is_empty() {
        peaks.push(0.0);
    }
    peaks
}

/// 按指定方式归一化描述符向量（就地修改）。
///
/// # Arguments
///
/// * `descriptor` - 待归一化的描述符向量
/// * `normalization` - 归一化方式：L1 直接除以 L1 范数；L2 先归一化、
///   再截断到 0.2、最后二次归一化
fn normalize_descriptor(descriptor: &mut [f32], normalization: Normalization) {
    match normalization {
        Normalization::L2 => {
            let norm: f32 = descriptor.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                for value in descriptor.iter_mut() {
                    *value /= norm;
                }
            }
            let max_value = 0.2;
            for value in descriptor.iter_mut() {
                if *value > max_value {
                    *value = max_value;
                }
            }
            let norm_after_clipping: f32 = descriptor.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_after_clipping > f32::EPSILON {
                for value in descriptor.iter_mut() {
                    *value /= norm_after_clipping;
                }
            }
        }
        Normalization::L1 => {
            let norm: f32 = descriptor.iter().map(|&x| x.abs()).sum();
            if norm > f32::EPSILON {
                for value in descriptor.iter_mut() {
                    *value /= norm;
                }
            }
        }
    }
}

/// 计算关键点的 SIFT 描述符。
///
/// 分三阶段执行：先在 GPU 上计算梯度；再计算方向直方图并读回 CPU 做峰值
/// 检测以展开关键点（一个关键点可能分裂为多个主方向）；最后在 GPU 上计算
/// 描述符并读回 CPU 归一化。
///
/// # Arguments
///
/// * `client` - cubecl 计算客户端
/// * `config` - 描述符配置参数
/// * `keypoints` - 待计算的关键点列表
/// * `pyramids` - 每个 octave 的高斯金字塔（保留在 GPU 显存中）
/// * `normalization` - 描述符归一化方式（L1 或 L2）
///
/// # Returns
///
/// `(带方向的关键点列表, 描述符列表)`，两者按索引一一对应。
pub(crate) fn compute_descriptors<R: Runtime>(
    client: &ComputeClient<R>,
    config: &DescriptorConfig,
    keypoints: Vec<KeyPoint>,
    pyramids: &[OctavePyramid],
    normalization: Normalization,
) -> (Vec<KeyPoint>, Vec<Vec<f32>>) {
    // 阶段一：GPU 计算每个八度的梯度幅值与方向（直接消费显存中的高斯金字塔）
    let mut octaves: Vec<OctaveGrad<R>> = Vec::with_capacity(pyramids.len());
    for pyramid in pyramids {
        let total = pyramid.num_scales * pyramid.height * pyramid.width;

        let mag = client.empty(total * core::mem::size_of::<f32>());
        let ori = client.empty(total * core::mem::size_of::<f32>());

        let cube_dim = CubeDim::new(client, total);
        let cube_count = calculate_cube_count_elemwise(client, total, cube_dim);

        unsafe {
            compute_gradients::launch::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(pyramid.gaussian.clone(), total),
                ArrayArg::from_raw_parts(mag.clone(), total),
                ArrayArg::from_raw_parts(ori.clone(), total),
                pyramid.width,
                pyramid.height,
            );
        }

        octaves.push(OctaveGrad {
            mag,
            ori,
            num_scales: pyramid.num_scales,
            height: pyramid.height,
            width: pyramid.width,
            _marker: core::marker::PhantomData,
        });
    }

    // 阶段二：GPU 计算方向直方图，CPU 做峰值检测并展开关键点
    let mut resolved: Vec<KeyPoint> = Vec::new();
    for (octave_idx, octave) in octaves.iter().enumerate() {
        let group: Vec<KeyPoint> = keypoints
            .iter()
            .filter(|k| k.octave == octave_idx)
            .cloned()
            .collect();

        if group.is_empty() {
            continue;
        }

        let n = group.len();
        let mut xs = vec![0.0f32; n];
        let mut ys = vec![0.0f32; n];
        let mut scales = vec![0.0f32; n];
        let mut octave_scales = vec![0.0f32; n];
        let mut scale_idx = vec![0u32; n];

        for (i, k) in group.iter().enumerate() {
            xs[i] = k.x;
            ys[i] = k.y;
            scales[i] = k.scale;
            octave_scales[i] = 2.0_f32.powi(k.octave as i32 + k.first_octave);
            scale_idx[i] = k.scale_idx as u32;
        }

        let h_x = client.create_from_slice(f32::as_bytes(&xs));
        let h_y = client.create_from_slice(f32::as_bytes(&ys));
        let h_scale = client.create_from_slice(f32::as_bytes(&scales));
        let h_octave_scale = client.create_from_slice(f32::as_bytes(&octave_scales));
        let h_scale_idx = client.create_from_slice(u32::as_bytes(&scale_idx));

        let hist = client.empty(n * config.num_bins * core::mem::size_of::<f32>());

        let cube_dim = CubeDim::new(client, n);
        let cube_count = calculate_cube_count_elemwise(client, n, cube_dim);

        let grad_len = octave.num_scales * octave.height * octave.width;

        unsafe {
            compute_orientation_histogram::launch::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(octave.mag.clone(), grad_len),
                ArrayArg::from_raw_parts(octave.ori.clone(), grad_len),
                ArrayArg::from_raw_parts(h_x, n),
                ArrayArg::from_raw_parts(h_y, n),
                ArrayArg::from_raw_parts(h_scale, n),
                ArrayArg::from_raw_parts(h_octave_scale, n),
                ArrayArg::from_raw_parts(h_scale_idx, n),
                ArrayArg::from_raw_parts(hist.clone(), n * config.num_bins),
                octave.width,
                octave.height,
                octave.height * octave.width,
                config.num_bins,
                config.gaussian_sigma_factor,
            );
        }

        let bytes = client.read_one(hist).unwrap();
        let hist_data = f32::from_bytes(&bytes);

        for i in 0..n {
            let h = &hist_data[i * config.num_bins..(i + 1) * config.num_bins];
            let peaks = find_peaks(h, config.num_bins, config.peak_ratio);
            for peak in peaks {
                let mut kp = group[i].clone();
                kp.orientation = peak;
                resolved.push(kp);
            }
        }
    }

    // 阶段三：GPU 计算描述符，CPU 归一化
    let desc_len = config.descriptor_width * config.descriptor_width * config.descriptor_num_bins;
    let mut descriptors: Vec<Vec<f32>> = Vec::with_capacity(resolved.len());

    for (octave_idx, octave) in octaves.iter().enumerate() {
        let group: Vec<&KeyPoint> = resolved.iter().filter(|k| k.octave == octave_idx).collect();
        if group.is_empty() {
            continue;
        }

        let n = group.len();
        let mut xs = vec![0.0f32; n];
        let mut ys = vec![0.0f32; n];
        let mut scales = vec![0.0f32; n];
        let mut octave_scales = vec![0.0f32; n];
        let mut scale_idx = vec![0u32; n];
        let mut orientations = vec![0.0f32; n];

        for (i, k) in group.iter().enumerate() {
            xs[i] = k.x;
            ys[i] = k.y;
            scales[i] = k.scale;
            octave_scales[i] = 2.0_f32.powi(k.octave as i32 + k.first_octave);
            scale_idx[i] = k.scale_idx as u32;
            orientations[i] = k.orientation;
        }

        let d_x = client.create_from_slice(f32::as_bytes(&xs));
        let d_y = client.create_from_slice(f32::as_bytes(&ys));
        let d_scale = client.create_from_slice(f32::as_bytes(&scales));
        let d_octave_scale = client.create_from_slice(f32::as_bytes(&octave_scales));
        let d_scale_idx = client.create_from_slice(u32::as_bytes(&scale_idx));
        let d_orientation = client.create_from_slice(f32::as_bytes(&orientations));

        let desc = client.empty(n * desc_len * core::mem::size_of::<f32>());

        let cube_dim = CubeDim::new(client, n);
        let cube_count = calculate_cube_count_elemwise(client, n, cube_dim);

        let grad_len = octave.num_scales * octave.height * octave.width;

        unsafe {
            compute_descriptor::launch::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(octave.mag.clone(), grad_len),
                ArrayArg::from_raw_parts(octave.ori.clone(), grad_len),
                ArrayArg::from_raw_parts(d_x, n),
                ArrayArg::from_raw_parts(d_y, n),
                ArrayArg::from_raw_parts(d_scale, n),
                ArrayArg::from_raw_parts(d_octave_scale, n),
                ArrayArg::from_raw_parts(d_scale_idx, n),
                ArrayArg::from_raw_parts(d_orientation, n),
                ArrayArg::from_raw_parts(desc.clone(), n * desc_len),
                octave.width,
                octave.height,
                octave.height * octave.width,
                config.descriptor_width,
                config.descriptor_num_bins,
            );
        }

        let bytes = client.read_one(desc).unwrap();
        let data = f32::from_bytes(&bytes);

        for i in 0..n {
            let start = i * desc_len;
            let mut d = data[start..start + desc_len].to_vec();
            normalize_descriptor(&mut d, normalization);
            descriptors.push(d);
        }
    }

    (resolved, descriptors)
}
