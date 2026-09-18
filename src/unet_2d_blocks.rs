use std::iter::zip;

use burn::{Tensor, config::Config, module::Module, tensor::backend::Backend};

use crate::{
    attention::{Attention, AttentionConfig},
    downsampling::{Downsample2D, Downsample2DConfig},
    resnet::{ResnetBlock2D, ResnetBlock2DConfig},
};

// Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py#L1372
#[derive(Module, Debug)]
pub struct DownEncoderBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    downsampler: Option<Downsample2D<B>>,
}

impl<B: Backend> DownEncoderBlock2D<B> {
    pub fn forward(&self, mut hidden_states: Tensor<B, 4>) -> Tensor<B, 4> {
        for resnet in &self.resnets {
            hidden_states = resnet.forward(hidden_states);
        }
        if let Some(downsampler) = &self.downsampler {
            hidden_states = downsampler.forward(hidden_states);
        }
        hidden_states
    }
}

// resnet_eps = 1e-6
// downsample_padding=0
#[derive(Config, Debug)]
pub struct DownEncoderBlock2DConfig {
    in_channels: usize,
    out_channels: usize,
    num_layers: usize,
    resnet_eps: f64,
    resnet_groups: usize,
    add_downsample: bool,
}

impl DownEncoderBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DownEncoderBlock2D<B> {
        let mut resnets = vec![];
        for i in 0..self.num_layers {
            // First layer expands the channels, every subsequent Resnet just keeps at out_channels
            let in_channels = if i == 0 {
                self.in_channels
            } else {
                self.out_channels
            };
            resnets.push(
                ResnetBlock2DConfig::new(
                    in_channels,
                    self.out_channels,
                    self.resnet_eps,
                    self.resnet_groups,
                )
                .init(device),
            );
        }

        DownEncoderBlock2D {
            resnets,
            downsampler: match self.add_downsample {
                true => {
                    Some(Downsample2DConfig::new(self.out_channels, self.out_channels).init(device))
                }
                false => None,
            },
        }
    }
}

// Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py#L589
#[derive(Module, Debug)]
pub struct UNetMidBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    attentions: Vec<Attention<B>>,
}

impl<B: Backend> UNetMidBlock2D<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden_states = self.resnets[0].forward(hidden_states);
        for (attn, resnet) in zip(&self.attentions, &self.resnets[1..]) {
            hidden_states = attn.forward(hidden_states);
            hidden_states = resnet.forward(hidden_states);
        }
        hidden_states
    }
}

// add_attention = true
// output_scale_factor=1
// num_layers = 1
// resnet_time_scale_shift = "default"

#[derive(Config, Debug)]
pub struct UNetMidBlock2DConfig {
    in_channels: usize,
    #[config(default = 1)]
    num_layers: usize,
    resnet_eps: f64,
    #[config(default = 512)]
    attention_head_dim: usize,
    resnet_groups: usize,
}

impl UNetMidBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNetMidBlock2D<B> {
        // There is always at least 1 resnet
        let mut resnets = vec![
            ResnetBlock2DConfig::new(
                self.in_channels,
                self.in_channels,
                self.resnet_eps,
                self.resnet_groups,
            )
            .init(device),
        ];
        let mut attentions = vec![];

        for _ in 0..self.num_layers {
            attentions.push(
                AttentionConfig::new(self.in_channels, self.resnet_eps, self.resnet_groups)
                    .init(device),
            );
            resnets.push(
                ResnetBlock2DConfig::new(
                    self.in_channels,
                    self.in_channels,
                    self.resnet_eps,
                    self.resnet_groups,
                )
                .init(device),
            );
        }
        UNetMidBlock2D {
            resnets,
            attentions,
        }
    }
}
