use imageproc::geometric_transformations::rotate_about_center;
use sift_cubecl::{KeyPoint, Sift, SiftParams};

fn visualize_matches(
    img1: &image::RgbImage,
    img2: &image::RgbImage,
    matches: &Vec<(KeyPoint, KeyPoint)>,
) -> image::RgbImage {
    let (w1, h1) = (img1.width(), img1.height());
    let (w2, h2) = (img2.width(), img2.height());

    // 创建拼接图像
    let mut combined_img = image::RgbImage::new(w1 + w2, h1.max(h2));

    // 复制第一张图像
    for y in 0..h1 {
        for x in 0..w1 {
            combined_img.put_pixel(x, y, *img1.get_pixel(x, y));
        }
    }

    // 复制第二张图像
    for y in 0..h2 {
        for x in 0..w2 {
            combined_img.put_pixel(x + w1, y, *img2.get_pixel(x, y));
        }
    }

    // 绘制匹配连线
    for (kp1, kp2) in matches {
        let x1 = kp1.x as u32;
        let y1 = kp1.y as u32;
        let x2 = kp2.x as u32 + w1; // 因为第二张图像在拼接图的右侧
        let y2 = kp2.y as u32;

        draw_line(&mut combined_img, x1, y1, x2, y2, image::Rgb([0, 255, 0]));
    }

    combined_img
}

fn draw_line(
    image: &mut image::RgbImage,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    color: image::Rgb<u8>,
) {
    let dx = x2 as i32 - x1 as i32;
    let dy = y2 as i32 - y1 as i32;

    let steps = dx.abs().max(dy.abs());

    if steps == 0 {
        return;
    }

    let x_inc = dx as f32 / steps as f32;
    let y_inc = dy as f32 / steps as f32;

    let mut x = x1 as f32;
    let mut y = y1 as f32;

    for _ in 0..=steps {
        if x < 0.0 || x >= image.width() as f32 || y < 0.0 || y >= image.height() as f32 {
            x += x_inc;
            y += y_inc;
            continue;
        }
        image.put_pixel(x as u32, y as u32, color);
        x += x_inc;
        y += y_inc;
    }
}

// 计算两个描述符之间的欧几里得距离
fn calculate_descriptor_distance(desc1: &[f32], desc2: &[f32]) -> f32 {
    if desc1.len() != desc2.len() {
        return f32::MAX;
    }

    let mut sum = 0.0;
    for i in 0..desc1.len() {
        let diff = desc1[i] - desc2[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

// 匹配两组关键点（使用Lowe's ratio test）
fn match_keypoints(
    kp1: &[KeyPoint],
    des1: &[Vec<f32>],
    kp2: &[KeyPoint],
    des2: &[Vec<f32>],
) -> Vec<(KeyPoint, KeyPoint)> {
    knn_match_keypoints(kp1, des1, kp2, des2, 2, Some(0.9), None)
}

// KNN匹配两组关键点
fn knn_match_keypoints(
    kp1: &[KeyPoint],
    des1: &[Vec<f32>],
    kp2: &[KeyPoint],
    des2: &[Vec<f32>],
    k: usize,
    ratio_threshold: Option<f32>,
    distance_threshold: Option<f32>,
) -> Vec<(KeyPoint, KeyPoint)> {
    let mut all_matches = Vec::new();

    for (i1, k1) in kp1.iter().enumerate() {
        if des1[i1].is_empty() {
            continue;
        }

        // 存储所有可能匹配及其距离和索引
        let mut candidates: Vec<(f32, usize)> = Vec::new();

        for (idx, _) in kp2.iter().enumerate() {
            if des2[idx].is_empty() {
                continue;
            }

            let dist = calculate_descriptor_distance(&des1[i1], &des2[idx]);
            candidates.push((dist, idx));
        }

        // 按距离排序
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 获取实际可用的匹配数量
        let valid_matches = candidates.len().min(k);

        if valid_matches == 0 {
            continue;
        }

        // 根据是否有ratio test来决定如何选择匹配
        if let Some(ratio) = ratio_threshold {
            // 使用Lowe's ratio test (通常用于SIFT)
            if candidates.len() >= 2 {
                let first_dist = candidates[0].0;
                let second_dist = candidates[1].0;

                // 检查距离阈值
                if let Some(max_dist) = distance_threshold {
                    if first_dist > max_dist {
                        continue;
                    }
                }

                // 应用ratio test
                if first_dist < ratio * second_dist {
                    all_matches.push((k1.clone(), kp2[candidates[0].1].clone()));
                }
            }
        } else {
            // 直接返回前K个匹配
            for i in 0..valid_matches {
                let dist = candidates[i].0;

                // 检查距离阈值
                if let Some(max_dist) = distance_threshold {
                    if dist > max_dist {
                        break; // 由于已排序，后续距离会更大
                    }
                }

                all_matches.push((k1.clone(), kp2[candidates[i].1].clone()));
            }
        }
    }

    all_matches
}

fn main() {
    // 加载原始图像
    let original_img = image::open("./examples/Lenna.jpg").unwrap();
    let gray_original = original_img.grayscale();

    // 旋转图像45度
    let rgb_img = original_img.to_rgb8();
    let angle_45 = std::f32::consts::FRAC_PI_8; // 22.5度 = π/8 弧度
    let rotated_img = rotate_about_center(
        &rgb_img,
        angle_45,
        imageproc::geometric_transformations::Interpolation::Bilinear,
        image::Rgb([0, 0, 0]),
    );

    // 保存一份RGB旋转图像的副本用于可视化
    let rgb_rotated = rotated_img.clone();

    // 将旋转后的RGB图像转换为灰度图用于特征检测
    let gray_rotated = image::DynamicImage::ImageRgb8(rotated_img).grayscale();

    let start = std::time::Instant::now();

    let params = SiftParams {
        first_octave: -1,
        ..SiftParams::default()
    };

    let sift = Sift::default();
    
    // 检测关键点并计算描述符
    let (keypoints1, descriptors1) = sift.detect_and_compute(gray_original, &params);
    println!("Found {} keypoints in original image", keypoints1.len());

    let (keypoints2, descriptors2) = sift.detect_and_compute(gray_rotated, &params);
    println!("Found {} keypoints in rotated image", keypoints2.len());

    println!("Running time: {:?}", start.elapsed());

    // 匹配关键点
    println!("Matching keypoints...");
    let matches = match_keypoints(&keypoints1, &descriptors1, &keypoints2, &descriptors2);
    println!("Found {} matches", matches.len());

    // 创建匹配可视化
    let rgb_original = original_img.to_rgb8();
    let matched_img = visualize_matches(&rgb_original, &rgb_rotated, &matches);

    // 保存结果
    matched_img.save("./examples/sift_matches.jpg").unwrap();
    println!("Matches visualization saved to ./examples/sift_matches.jpg");}
