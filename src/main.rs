use crate::vae::VAEConfig;
use burn::{Tensor, backend::Wgpu};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use image::{GenericImageView, Rgb, RgbImage};

mod attention;
mod decoder;
mod downsampling;
mod encoder;
mod resnet;
mod unet_2d_blocks;
mod upsampling;
mod vae;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = Default::default();
    let block_out_channels = vec![128, 256, 512, 512];
    let mut vae = VAEConfig::new(block_out_channels).init::<MyBackend>(&device);

    let remapper = KeyRemapper::new()
        .add_pattern(r"\.to_out\.0\.", ".to_out.")
        .unwrap();

    let mut store = SafetensorsStore::from_file("vae_f32.safetensors")
        .remap(remapper)
        .with_from_adapter(PyTorchToBurnAdapter);

    let result = vae.load_from(&mut store).unwrap();
    println!("{:?}\n\n", vae);
    println!("{}", result);

    println!("Applied: {} tensors", result.applied.len());
    assert_eq!(result.applied.len(), 244, "Weights didn't load");
    println!("Missing: {:?}", result.missing);
    println!("Errors: {:?}", result.errors);

    if result.is_success() {
        println!("All tensors loaded successfully");
    }

    let img = image::open("image.png").expect("Failed to open image");

    let (width, height) = img.dimensions();

    let mut pixels = Vec::with_capacity(3 * width as usize * height as usize);
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let p = img.get_pixel(x, y).0[c];
                pixels.push((p as f32 / 127.5) - 1.0);
            }
        }
    }

    let input = Tensor::<MyBackend, 1>::from_floats(pixels.as_slice(), &device)
        .reshape([1, 3, height, width]);

    let output = vae.forward(input.clone());
    let [_b, _c, h, w] = output.dims();

    let out_data = output
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("Failed to convert tensor to vector");

    let hw = h * w;
    let mut reconstructed = RgbImage::new(w as u32, h as u32);

    // Map NCHW into interleaved RGB pixels
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let r = ((out_data[0 * hw + idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let g = ((out_data[1 * hw + idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let b = ((out_data[2 * hw + idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            reconstructed.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    reconstructed
        .save("reconstructed.png")
        .expect("Failed to save reconstructed image");
}
