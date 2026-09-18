// Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/downsampling.py
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{backend::Backend, ops::PadMode},
};

#[derive(Module, Debug)]
pub struct Downsample2D<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> Downsample2D<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 4>) -> Tensor<B, 4> {
        let hidden_states = hidden_states.pad((0, 1, 0, 1), PadMode::Constant(0.0));
        self.conv.forward(hidden_states)
    }
}

// padding = 0, cascaded down from vae.py
// use_conv = true
#[derive(Config, Debug)]
pub struct Downsample2DConfig {
    channels: usize,
    out_channels: usize,
}

impl Downsample2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Downsample2D<B> {
        Downsample2D {
            conv: Conv2dConfig::new([self.channels, self.out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                .init(device),
        }
    }
}
