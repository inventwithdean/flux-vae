use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend},
};

// Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    group_norm: GroupNorm<B>,
    to_k: Linear<B>,
    to_q: Linear<B>,
    to_v: Linear<B>,
    to_out: Linear<B>,
}

impl<B: Backend> Attention<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 4>) -> Tensor<B, 4> {
        let input_tensor = hidden_states.clone();
        let hidden_states = self.group_norm.forward(hidden_states); // (B, C, H, W)
        let [b, c, h, w] = hidden_states.dims();

        let hidden_states: Tensor<B, 3> = hidden_states.flatten(2, 3); // (B, C, H*W)

        let hidden_states = hidden_states.swap_dims(1, 2); // (B, H*W, C)

        // (H*W, 512) @ (512, 512) = (H*W, 512)
        let queries = self.to_q.forward(hidden_states.clone());
        let keys = self.to_k.forward(hidden_states.clone());
        let values = self.to_v.forward(hidden_states.clone());
        // Q, K and V are of the shape (B, H*W, 512)
        let d_k: f64 = keys.shape()[2] as f64; // 512

        let keys_transposed = keys.transpose(); // (B, 512, H*W)
        // Perform attention
        // (Q @ K^T) @ V / sqrt(d_k)
        let mut scores = queries.matmul(keys_transposed) / d_k.sqrt(); // (B, H*W, H*W)
        // Let's assume H was 16 and W was 16, so H*W was 256
        // (B, 256, 256), so first row of first batch item, i.e. [0, 0, :] represents all the weights (query key dot product) which means how much other 256 patches are of importance to this patch.
        scores = softmax(scores, 2); // (B, H*W, H*W) with last_dim now being normalized to sum to 1

        // (B, H*W, H*W) @ (B, H*W, 512)
        let attention = scores.matmul(values); // (B, H*W, 512)
        // Output projection: (B, H*W, 512) @ (512, 512) => (B, H*W, 512)
        let attention = self.to_out.forward(attention); // (B, H*W, 512)

        let attention = attention.transpose(); // (B, 512, H*W)
        let attention = attention.reshape([b, c, h, w]);
        attention + input_tensor
    }
}

// query_dim = in_channels = 512
// spatial_norm_dim = None
// residual_connection = true
// bias = true
// 📋 upcast_softmax = true
// _from_deprecated_attn_block = true
// eps = 1e-6
// heads = in_channels // attention_head_dim = 512 // 512 = 1
#[derive(Config, Debug)]
pub struct AttentionConfig {
    in_channels: usize,
    eps: f64,
    norm_num_groups: usize,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        Attention {
            group_norm: GroupNormConfig::new(self.norm_num_groups, self.in_channels)
                .with_epsilon(self.eps)
                .init(device),
            to_k: LinearConfig::new(self.in_channels, self.in_channels).init(device),
            to_q: LinearConfig::new(self.in_channels, self.in_channels).init(device),
            to_v: LinearConfig::new(self.in_channels, self.in_channels).init(device),
            to_out: LinearConfig::new(self.in_channels, self.in_channels).init(device),
        }
    }
}
