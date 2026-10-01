use sift_cubecl::{KeyPoint, Sift, SiftParams};

fn visualize_keypoints(image: &mut image::RgbImage, keypoints: &[KeyPoint]) {
    for kp in keypoints {
        let x = kp.x as u32;
        let y = kp.y as u32;

        // 绘制关键点位置
        for dy in -2..=2 {
            for dx in -2..=2 {
                let px = (x as i32 + dx).max(0).min(image.width() as i32 - 1) as u32;
                let py = (y as i32 + dy).max(0).min(image.height() as i32 - 1) as u32;
                //随机颜色
                image.put_pixel(px, py, image::Rgb([255, 0, 0]));
            }
        }

        // 绘制方向
        let length = 10.0;
        let end_x = x as f32 + length * kp.orientation.cos();
        let end_y = y as f32 + length * kp.orientation.sin();

        draw_line(
            image,
            x,
            y,
            end_x as u32,
            end_y as u32,
            image::Rgb([0, 255, 0]),
        );
    }
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
            continue;
        }
        image.put_pixel(x as u32, y as u32, color);
        x += x_inc;
        y += y_inc;
    }
}

fn main() {
    // 加载图像并转换为灰度图像
    let img = image::open("./examples/Lenna.jpg").unwrap();
    let gray_img = img.grayscale();
    let sift = Sift::default();
    let params = SiftParams {
        first_octave: -1,
        ..SiftParams::default()
    };
    let (keypoints, _descriptors) = sift.detect_and_compute(gray_img, &params);
    print!("{} keypoints found", keypoints.len());
    // // 可视化关键点
    let mut rgb_image = image::open("./examples/Lenna.jpg").unwrap().to_rgb8();
    visualize_keypoints(&mut rgb_image, &keypoints);
    rgb_image
        .save("./examples/output_with_keypoints.jpg")
        .unwrap();
}
