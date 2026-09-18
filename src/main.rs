use std::time::Instant;

use crate::encoder::EncoderConfig;
use burn::{Tensor, backend::Wgpu};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use image::GenericImageView;

mod attention;
mod downsampling;
mod encoder;
mod resnet;
mod unet_2d_blocks;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = Default::default();
    let block_out_channels = vec![128, 256, 512, 512];
    let mut encoder = EncoderConfig::new(block_out_channels).init::<MyBackend>(&device);

    let remapper = KeyRemapper::new()
        .add_pattern(r"^encoder\.", r"")
        .unwrap()
        .add_pattern(r"\.downsamplers\.0\.", ".downsampler.")
        .unwrap()
        .add_pattern(r"\.to_out\.0\.", ".to_out.")
        .unwrap();

    let mut store = SafetensorsStore::from_file("vae_f32.safetensors")
        // .with_regex(r"^encoder\..*")
        .remap(remapper)
        .with_from_adapter(PyTorchToBurnAdapter);

    let result = encoder.load_from(&mut store).unwrap();
    println!("{:?}\n\n", encoder);
    println!("{}", result);

    println!("Applied: {} tensors", result.applied.len());
    assert_eq!(result.applied.len(), 106, "Encoder weights didn't load");
    println!("Missing: {:?}", result.missing);
    println!("Errors: {:?}", result.errors);

    if result.is_success() {
        println!("All tensors loaded successfully");
    }

    let img = image::open("image.png")
        .expect("Failed to open dean.webp")
        .resize_exact(768, 768, image::imageops::FilterType::Lanczos3);

    let mut pixels = Vec::with_capacity(3 * 768 * 768);
    for c in 0..3 {
        for y in 0..768 {
            for x in 0..768 {
                let p = img.get_pixel(x, y).0[c];
                pixels.push((p as f32 / 127.5) - 1.0);
            }
        }
    }
    let input =
        Tensor::<MyBackend, 1>::from_floats(pixels.as_slice(), &device).reshape([1, 3, 768, 768]);

    let mut output = encoder.forward(input.clone());
    for _ in 0..2 {
        output = encoder.forward(input.clone());
    }

    const N: usize = 10;
    let mut total = std::time::Duration::ZERO;
    for _ in 0..N {
        let start = Instant::now();
        output = encoder.forward(input.clone());
        // let _ = output.clone().into_data();
        total += start.elapsed();
    }
    println!("Avg. over {} runs {:?}", N, total / N as u32);

    println!("Output shape: {:?}", output.dims());

    let out_data = output.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    println!("Rust Encoder (First 10): {:.4?}", &out_data[..10]);
    let bytes: &[u8] = bytemuck::cast_slice(&out_data);
    std::fs::write("rust_latents.bin", bytes).expect("Failed to write latents");
    println!("Latents successfully saved to rust_latents.bin!");
}
