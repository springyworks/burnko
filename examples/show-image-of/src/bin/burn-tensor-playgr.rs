#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
// Import burn packages
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::Distribution;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::prelude::*;
use ndarray::Array;
use std::path::Path;
// Type alias for the backend
type Backend = NdArray<f32>; //backend on cpu

fn main() {
    let img_path = file!();
    println!("Start of: {:?}", img_path); //print name of this source file.

    let device = Default::default();
    // Creation of two tensors, the first with explicit values
    // and the second one with ones, with the same shape as the first
    // let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3., 6.], [4., 5.,7.]], &device);
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2, 4, 8], [16, 32, 64]], &device);
    print!("\n{:?}\n", tensor_1.to_data().dtype);

    let tensor_int = tensor_1.int();
   print!("\n{:?\n}", tensor_int.to_data().dtype);
   
   
// //let td = &tensor_1.to_data();
// print!("{:?}", td);

// let width = td.shape[1] as u32;
// let height = td.shape[0] as u32;
// let dtype = td.dtype;

// let bytes: Vec<u8> = match dtype {
//     DType::U8 => td.bytes.clone(),
//     DType::F32 => td.bytes.iter().map(|&x| (x as f32 * 255.0) as u8).collect(),
//     // Add more cases as needed for other data types
//     _ => panic!("Unsupported tensor data type"),
// };

// // Create an RgbImage from the bytes
// let image = RgbImage::from_vec(width, height, bytes).unwrap();

// // Save or display the image as needed
// image.save("output.png").unwrap();
}
