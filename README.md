# sift_cubecl

Scale-Invariant Feature Transform (SIFT) 的 Rust 实现，使用 [cubecl](https://github.com/tracel-ai/cubecl) 加速。

支持关键点检测（detection）与 128 维描述符（descriptor）计算，可运行在 CPU、wgpu（Vulkan/Metal/DX12）与 NVIDIA CUDA 后端。

## 特性

- 完整 SIFT 流程：高斯金字塔 → DoG → 极值检测 → 亚像素精化 → 边缘响应剔除 → 方向直方图 → 128 维描述符
- 多 octave 尺度空间，支持 `first_octave = -1` 上采样
- 描述符归一化支持 L1 / L2
- 通过 cubecl 在 CPU / GPU 上执行，计算与数据搬移统一走显存

## 后端（feature）

| feature | 后端 | 运行时依赖 | 适用场景 |
|---------|------|-----------|---------|
| `wgpu`  | Vulkan / Metal / DX12 | 无 | 跨平台，默认 |
| `cuda`  | NVIDIA CUDA | CUDA Toolkit + 匹配驱动 | NVIDIA 卡 |
| `cpu`   | cubecl CPU 运行时 | Windows 需匹配的 MSVC | 无 GPU 环境 |

默认 feature 为 `wgpu`。三选一即可。

## 快速开始

```toml
# 你项目的 Cargo.toml
[dependencies]
sift-cubecl = { path = "../sift_cubecl" }   # 默认 wgpu
image = "0.25"
```

```rust
use sift_cubecl::{Sift, SiftParams};

fn main() {
    let sift = Sift::default();
    let params = SiftParams {
        first_octave: -1,                 // 上采样 2x，检测更多关键点
        ..SiftParams::default()
    };

    let img = image::open("lena.jpg").unwrap().grayscale();

    // 只检测关键点
    let keypoints = sift.detect(img.clone(), &params);

    // 检测 + 计算描述符（128 维）
    let (keypoints, descriptors) = sift.detect_and_compute(img, &params);

    println!("{} keypoints", keypoints.len());
    println!("descriptor dim = {}", descriptors[0].len());
}
```

## 在依赖方切换后端

推荐用 **feature 转发**，让下游项目拥有自己的 feature：

```toml
[features]
default = ["wgpu"]
cpu   = ["sift-cubecl/cpu"]
cuda  = ["sift-cubecl/cuda"]
wgpu  = ["sift-cubecl/wgpu"]

[dependencies]
sift-cubecl = { path = "../sift_cubecl", default-features = false }
image = "0.25"
```

```bash
cargo run                                       # wgpu（默认）
cargo run --no-default-features --features cpu   # cpu
cargo run --no-default-features --features cuda  # cuda
```

`Sift` 类型别名由 feature 在编译期决定，切换 feature 后重编译即可，代码零改动。

### 同时启用多个后端（运行时切换）

多个 feature 会编译进**同一个二进制**，用泛型 `Sift<R>` 在运行时选择：

```toml
features = ["cpu", "cuda", "wgpu"]
```

```rust
use sift_cubecl::sift::Sift;
use sift_cubecl::SiftParams;

fn run<R: cubecl::Runtime>(device: R::Device, img: image::DynamicImage, params: &SiftParams)
where
    R::Device: Default,
{
    let sift = Sift::<R>::new(device);
    let (kp, _) = sift.detect_and_compute(img, params);
    println!("{} keypoints", kp.len());
}

fn main() {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "wgpu".into());
    let img = image::open("lena.jpg").unwrap().grayscale();
    let params = SiftParams::default();

    match backend.as_str() {
        "cpu"  => run::<cubecl::cpu::CpuRuntime>(cubecl::cpu::CpuDevice::default(), img, &params),
        "cuda" => run::<cubecl::cuda::CudaRuntime>(cubecl::cuda::CudaDevice::default(), img, &params),
        _      => run::<cubecl::wgpu::WgpuRuntime>(cubecl::wgpu::WgpuDevice::default(), img, &params),
    }
}
```

## API

### `Sift`

| 方法 | 说明 |
|------|------|
| `Sift::default()` | 用所选后端的默认设备创建 |
| `Sift::new(device)` | 用指定设备创建 |
| `detect(image, params)` | 返回 `Vec<KeyPoint>` |
| `detect_and_compute(image, params)` | 返回 `(Vec<KeyPoint>, Vec<Vec<f32>>)`，描述符为 128 维 |

输入为 `image::DynamicImage`（灰度，内部仅接受 `ImageLuma8`，其他格式请先 `.grayscale()`）。

### `SiftParams`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `max_num_features` | `0` | 最多保留关键点数，`0` 表示不限制 |
| `first_octave` | `0` | 首个 octave 索引，`-1` 表示先上采样 2x |
| `num_octaves` | `0` | octave 数，`0` 表示按图像尺寸自动 |
| `octave_resolution` | `3` | 每个 octave 的尺度数 |
| `peak_threshold` | `0.04` | 极值响应阈值（越小检测越多） |
| `edge_threshold` | `10.0` | 边缘响应剔除阈值 |
| `normalization` | `L1` | 描述符归一化：`Normalization::L1` / `L2` |

### `KeyPoint`

```rust
pub struct KeyPoint {
    pub x: f32,           // 关键点 x 坐标
    pub y: f32,           // 关键点 y 坐标
    pub scale: f32,       // 实际尺度 sigma
    pub octave: usize,    // 所在 octave
    pub first_octave: i32,// 首 octave 索引
    pub orientation: f32, // 主方向（弧度）
    pub scale_idx: usize, // 高斯金字塔层索引
}
```

## 运行示例

```bash
# 关键点可视化（输出 output_with_keypoints.jpg）
cargo run --example test

# 图像匹配（输出 sift_matches.jpg）
cargo run --example sift_match
```

需在 `examples/` 下放置 `Lenna.jpg`。

## 后端环境要求

- **wgpu**：无需额外安装（Vulkan/Metal/DX12）。
- **cuda**：安装 CUDA Toolkit，并设置环境变量：
  ```
  CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<版本>
  PATH     += ...\CUDA\<版本>\bin 和 ...\CUDA\<版本>\bin\x64
  ```
  且 NVIDIA 驱动版本需匹配 Toolkit（如 Toolkit 13.x 需驱动 ≥ 580）。
- **cpu**：Windows 下需 Visual Studio 2022 Build Tools（MSVC 14.42+），用于 cubecl 的 MLIR/LLVM 后端。

## Kernel 编译缓存

cubecl 会把编译好的 kernel 缓存到磁盘，跨进程复用（cuda/wgpu 支持，cpu 不支持）。在**运行时的项目根目录**放置 `cubecl.toml`：

```toml
[compilation]
cache = "target"   # 或 "global"
```

第一次运行仍会编译，之后命中缓存跳过编译。

## License

MIT OR Apache-2.0
