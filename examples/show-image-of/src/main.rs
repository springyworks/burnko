#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
// Import burn packages
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
// Import plotting library
//use evcxr_image::ImageDisplay;
//use image::save_buffer_with_format;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

// Type alias for the backend
type B = NdArray<f32>;

fn main() {
    println!("Start of printing image of tensor");
    let device = Default::default();
    // Create a random tensor
    let tensor_r: Tensor<B, 3> = Tensor::random([3, 32, 32], Distribution::Default, &device);
    let tensor_rs = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([2, 2]);
    let tensor_bool = Tensor::<B, 1, Bool>::from_data([1, 0, 1, 1], &device);

    // let tensor_1 = Tensor::<B, 3>::from_data([[20, 30], [40, 50]], &device);
    // To use B,3 correctly, ensure that the dimensions and the type match the requirements for a 3-dimensional tensor.
    // Assuming B is a valid type for the Tensor and the device is correctly defined,
    // the data provided must be adjusted to fit a 3-dimensional structure.

    let tensor_1 = Tensor::<B, 3, Int >::from_data([[[20], [30]], [[40], [50]]], &device);
    let tensordata = tensor_1.to_data().bytes;
    println!("tensor_data = {:?}", &tensordata);

   
    let img_buff = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(2, 2, tensordata).unwrap();

    img_buff.save("image_dirext-of-tensor.jpg").unwrap();

    // img_buff
    // .save_with_format("image-of-tensor.tiff", image::ImageFormat::Tiff)
    // .unwrap();

    print!("\nEnd of printing image of tensor\n");
}

//let img_dyn = DynamicImage::ImageRgb8(img_buff);
//img_dyn.save("randomtensorimg.jpg").unwrap();
//save_buffer_with_format("myimg.jpg", &tensordata.bytes, 32, 32, image::ColorType::Rgb8, image::ImageFormat::Jpeg).unwrap();

// let mut img = RgbImage::new(32, 32);
// for x in 15..=17 {
//     for y in 8..24 {
//         img.put_pixel(x, y, Rgb([255, 0, 0]));
//         img.put_pixel(y, x, Rgb([255, 0, 0]));
//     }
// }
// img.save("./test.png").unwrap();

// // Create an image buffer
// // TODO Use tenso to display plots
// let imag=ImageBuffer::from_fn(256, 256, |x, y| {
//     if (x as i32 - y as i32).abs() < 3 {
//         image::Rgb([0, 0, 255])
//     } else {
//         image::Rgb([0, 0, 0])
//     }
// });
