#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
// Import burn packages
use burn::prelude::*;
use burn_ndarray::NdArray;
use burn::backend::Wgpu;

// Type alias for the backend to use.
//type Backend = Wgpu;

// Type alias for the backend
type Backend = NdArray<f32>; 

fn main() {
    println!("Hello, world!");
    let device=Default::default();
  
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);}
