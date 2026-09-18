use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        backend::Backend,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

// Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/upsampling.py#L74
#[derive(Module, Debug)]
pub struct Upsample2D<B: Backend> {
    conv: Conv2d<B>,
}

// scale_factor = 2
impl<B: Backend> Upsample2D<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_b, _c, h, w] = hidden_states.dims();
        let hidden_states = interpolate(
            hidden_states,
            [h * 2, w * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );

        self.conv.forward(hidden_states)
    }
}

// use_conv = true
// padding = 1
#[derive(Config, Debug)]
pub struct Upsample2DConfig {
    channels: usize,
    out_channels: usize,
}

impl Upsample2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Upsample2D<B> {
        Upsample2D {
            conv: Conv2dConfig::new([self.channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }
}
