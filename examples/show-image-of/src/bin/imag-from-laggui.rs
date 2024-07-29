use burn::{
    // backend::NdArray, // Remove or comment out this line
    tensor::{backend::Backend, Device, Element, Shape, Tensor, TensorData},
};
use burn_ndarray::NdArray;
use image::{ImageBuffer, RgbImage};
use std::path::Path;

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)).convert::<f32>(), device)
        // permute(2, 0, 1)
        .swap_dims(2, 1) // [H, C, W]
        .swap_dims(1, 0) // [C, H, W]
        / 255 // normalize between [0, 1]
}

fn to_image(
    tensor: Tensor<NdArray, 4>,
    height: u32,
    width: u32,
) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    // Assuming the tensor is in [B, C, H, W] format and we're dealing with the first image in the batch
    let tensor_data = tensor.to_data().bytes; // Remove batch? dimension and get raw data

    let img_buff =
        ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(height, width, tensor_data).unwrap();
   return img_buff;
}

type Bckend = NdArray<f32>;
fn main() {
    let device = Default::default();

    let img_path = Path::new(file!())
        .parent()
        .unwrap()
        .join("../../assets/tinyjpg-in.jpg");
    println!("Image path: {:?}", img_path);
    let img = image::open(img_path).unwrap();

    // Resize to WIDTH x HEIGHT
    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle, // also known as bilinear in 2D
    );

    // Create tensor from image data
    let img_tensor: Tensor<Bckend, 4> = to_tensor(
        resized_img.into_rgb8().into_raw(),
        [HEIGHT, WIDTH, 3],
        &device,
    )
    .unsqueeze::<4>(); // [B, C, H, W]

    // Usage example
    let tensor_image: RgbImage = to_image(img_tensor, HEIGHT as u32, WIDTH as u32);
    tensor_image.save("image.jpg").unwrap();
    // Normalize the image
    //let x = imagenet::Normalizer::new(&device).normalize(img_tensor);
}
