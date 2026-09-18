use std::vec;

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
    UNetMidBlock2D, UNetMidBlock2DConfig, UpDecoderBlock2D, UpDecoderBlock2DConfig,
};

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    conv_in: Conv2d<B>,
    up_blocks: Vec<UpDecoderBlock2D<B>>,
    mid_block: UNetMidBlock2D<B>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, sample: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut sample = self.conv_in.forward(sample);
        sample = self.mid_block.forward(sample);
        for up_block in &self.up_blocks {
            sample = up_block.forward(sample);
        }
        sample = self.conv_norm_out.forward(sample);
        sample = silu(sample);
        self.conv_out.forward(sample)
    }
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    #[config(default = 16)]
    in_channels: usize,
    #[config(default = 3)]
    out_channels: usize,
    block_out_channels: Vec<usize>,
    #[config(default = 2)]
    layers_per_block: usize,
    #[config(default = 32)]
    norm_num_groups: usize,
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        // 128
        let block_out_channel_first = self
            .block_out_channels
            .first()
            .expect("block_out_channels should not be empty!")
            .to_owned();
        // 512
        let block_out_channel_last = self
            .block_out_channels
            .last()
            .expect("block_out_channels should not be empty!")
            .to_owned();

        let mut up_blocks = vec![];
        let mut output_channel = block_out_channel_last; // 512
        let num_up_blocks = self.block_out_channels.len();

        // Original: 128, 256, 512, 512
        // Reversed: 512, 512, 256, 128
        let block_out_channels_reversed: Vec<usize> =
            self.block_out_channels.iter().rev().cloned().collect();

        for i in 0..num_up_blocks {
            let prev_output_channel = output_channel;
            output_channel = block_out_channels_reversed[i];
            let is_final_block = i == block_out_channels_reversed.len() - 1;
            up_blocks.push(
                UpDecoderBlock2DConfig::new(
                    prev_output_channel,
                    output_channel,
                    self.layers_per_block + 1,
                    1e-6,
                    self.norm_num_groups,
                    !is_final_block,
                )
                .init(device),
            );
        }

        Decoder {
            conv_in: Conv2dConfig::new([self.in_channels, block_out_channel_last], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            up_blocks,
            mid_block: UNetMidBlock2DConfig::new(
                block_out_channel_last,
                1e-6,
                self.norm_num_groups,
            )
            .init(device),
            conv_norm_out: GroupNormConfig::new(self.norm_num_groups, block_out_channel_first)
                .with_epsilon(1e-6)
                .init(device),
            conv_out: Conv2dConfig::new([block_out_channel_first, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }
}
