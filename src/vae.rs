use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::{backend::Backend, s},
};

use crate::{
    decoder::{Decoder, DecoderConfig},
    encoder::{Encoder, EncoderConfig},
};

#[derive(Module, Debug)]
pub struct VAE<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> VAE<B> {
    pub fn forward(&self, input_tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let enc = self.encoder.forward(input_tensor);
        println!("enc Shape: {:?}", enc.dims());
        let mean = enc.slice(s![.., 0..16, .., ..]);
        println!("mean Shape: {:?}", mean.dims());
        let dec = self.decoder.forward(mean);
        println!("dec Shape: {:?}", dec.dims());
        dec
    }
}

#[derive(Config, Debug)]
pub struct VAEConfig {
    block_out_channels: Vec<usize>,
}

impl VAEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VAE<B> {
        VAE {
            encoder: EncoderConfig::new(self.block_out_channels.clone()).init(device),
            decoder: DecoderConfig::new(self.block_out_channels.clone()).init(device),
        }
    }
}
