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

use crate::unet_2d_blocks::{
    DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig,
};

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv_in: Conv2d<B>,
    down_blocks: Vec<DownEncoderBlock2D<B>>,
    mid_block: UNetMidBlock2D<B>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, sample: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut sample = self.conv_in.forward(sample);
        for down_block in &self.down_blocks {
            sample = down_block.forward(sample);
        }
        sample = self.mid_block.forward(sample);

        // Post processing
        sample = self.conv_norm_out.forward(sample);
        sample = silu(sample);
        self.conv_out.forward(sample)
    }
}

// double_z = true
// mid_block_add_attention = true
#[derive(Config, Debug)]
pub struct EncoderConfig {
    #[config(default = 3)]
    in_channels: usize,
    #[config(default = 16)]
    out_channels: usize,
    block_out_channels: Vec<usize>,
    #[config(default = 2)]
    layers_per_block: usize,
    #[config(default = 32)]
    norm_num_groups: usize,
    #[config(default = 4)]
    num_down_blocks: usize,
    #[config(default = 4)]
    num_up_blocks: usize,
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let mut down_blocks = vec![];
        let mut output_channel = self.block_out_channels[0];
        for i in 0..self.num_down_blocks {
            let input_channel = output_channel;
            output_channel = self.block_out_channels[i];
            let is_final_block = i == self.block_out_channels.len() - 1;

            down_blocks.push(
                DownEncoderBlock2DConfig::new(
                    input_channel,
                    output_channel,
                    self.layers_per_block,
                    1e-6,
                    self.norm_num_groups,
                    !is_final_block,
                )
                .init(device),
            );
        }

        let last_block_out_size = self
            .block_out_channels
            .last()
            .expect("block_out_channels should not be empty!")
            .to_owned();
        Encoder {
            conv_in: Conv2dConfig::new([self.in_channels, self.block_out_channels[0]], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            down_blocks,
            mid_block: UNetMidBlock2DConfig::new(last_block_out_size, 1e-6, self.norm_num_groups)
                .init(device),
            conv_norm_out: GroupNormConfig::new(self.norm_num_groups, last_block_out_size)
                .with_epsilon(1e-6)
                .init(device),
            conv_out: Conv2dConfig::new([last_block_out_size, 2 * self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }
}
