// Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        GroupNorm, GroupNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{activation::silu, backend::Backend},
};

#[derive(Module, Debug)]
pub struct ResnetBlock2D<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    conv_shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> ResnetBlock2D<B> {
    pub fn forward(&self, mut input_tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden_states = input_tensor.clone();
        hidden_states = self.norm1.forward(hidden_states);
        hidden_states = silu(hidden_states);
        hidden_states = self.conv1.forward(hidden_states);
        hidden_states = self.norm2.forward(hidden_states);
        hidden_states = silu(hidden_states);
        hidden_states = self.conv2.forward(hidden_states);
        if let Some(conv_shortcut) = &self.conv_shortcut {
            input_tensor = conv_shortcut.forward(input_tensor);
        }
        input_tensor + hidden_states
    }
}

// time_embedding_norm = "default"
// non_linearity = "silu"
// output_scale_factor = 1.0
// pre_norm = true
#[derive(Config, Debug)]
pub struct ResnetBlock2DConfig {
    in_channels: usize,
    out_channels: usize,
    eps: f64,
    groups: usize,
}

impl ResnetBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResnetBlock2D<B> {
        let use_in_shortcut = self.in_channels != self.out_channels;
        ResnetBlock2D {
            norm1: GroupNormConfig::new(self.groups, self.in_channels)
                .with_epsilon(self.eps)
                .init(device),
            conv1: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            norm2: GroupNormConfig::new(self.groups, self.out_channels)
                .with_epsilon(self.eps)
                .init(device),
            conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv_shortcut: match use_in_shortcut {
                true => Some(
                    Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                        .with_bias(true)
                        .init(device),
                ),
                false => None,
            },
        }
    }
}
