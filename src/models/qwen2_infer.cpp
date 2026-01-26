#include "qwen2.hpp"
#include "../tensor/tensor.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/add/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/linear/op.hpp"
#include <cmath>

namespace llaisys {

void Qwen2Model::forwardEmbedding(const tensor_t input_ids) {
  llaisys::ops::embedding(internal_buffers_.embed_out, input_ids, weights_.in_embed);
}

void Qwen2Model::forwardRMSNorm(const tensor_t input, const tensor_t weight,
                            tensor_t output, float epsilon) {
  llaisys::ops::rms_norm(output, input, weight, epsilon);
}

void Qwen2Model::forwardAttention(tensor_t input, int layer) {
  auto& buf = internal_buffers_;
  size_t seq_len = input->shape()[0];
  auto q_out = buf.q_out->slice(0, 0, seq_len);
  auto k_out = kvcache_.k[layer]->slice(0, 0, seq_len);
  auto v_out = kvcache_.v[layer]->slice(0, 0, seq_len);
  // q
  llaisys::ops::linear(q_out,
                       input,
                       weights_.attn_q_w[layer],
                       weights_.attn_q_b[layer]);
  int pos_id_slice_end = is_decode_stage ? 1 : seq_len;
  auto pos_ids = internal_buffers_.pos_id->slice(0, 0, pos_id_slice_end);
  llaisys::ops::rope(q_out, q_out, pos_ids, meta_.theta);
  // k
  llaisys::ops::linear(k_out,
                       input,
                       weights_.attn_k_w[layer],
                       weights_.attn_k_b[layer]);
  llaisys::ops::rope(k_out, k_out, pos_ids, meta_.theta);
  // v
  llaisys::ops::linear(v_out,
                       input,
                       weights_.attn_v_w[layer],
                       weights_.attn_v_b[layer]);
  
  float scale = 1.0f / sqrtf(static_cast<float>(meta_.dh));
  llaisys::ops::self_attention(internal_buffers_.attn_out,
                               internal_buffers_.s,
                               q_out,
                               k_out,
                               v_out,
                               scale);
}


int64_t Qwen2Model::infer(const int64_t* token_ids, size_t ntoken) {
    
    return -1;
}

} // namespace llaisys