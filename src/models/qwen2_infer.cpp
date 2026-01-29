#include "qwen2.hpp"
#include "../tensor/tensor.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/add/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/argmax/op.hpp"
#include <cmath>

namespace llaisys {

tensor_t Qwen2Model::forwardEmbedding(const tensor_t input_ids) {
  auto embed_out = internal_buffers_.embed_out->slice(0, 0, input_ids->shape()[0]);
  llaisys::ops::embedding(embed_out, input_ids, weights_.in_embed);
  return embed_out;
}

tensor_t Qwen2Model::forwardRMSNorm(const tensor_t input, const tensor_t weight, float epsilon) {
  auto output = internal_buffers_.rms_norm_out->slice(0, 0, input->shape()[0]);
  llaisys::ops::rms_norm(output, input, weight, epsilon);
  return output;
}

tensor_t Qwen2Model::forwardAttention(tensor_t input, size_t layer) {
  auto& buf = internal_buffers_;
  size_t seq_len = input->shape()[0];
  size_t pid = (size_t)pos_id;
  auto q_out = buf.q_out->slice(0, 0, seq_len);
  auto k_cache_out = kvcache_.k[layer]->slice(0, pid, pid + seq_len);
  auto v_cache_out = kvcache_.v[layer]->slice(0, pid, pid + seq_len);
  auto k_out = kvcache_.k[layer]->slice(0, 0, pid + seq_len);
  auto v_out = kvcache_.v[layer]->slice(0, 0, pid + seq_len);
  auto attn_out = buf.attn_out->slice(0, 0, seq_len);
  auto o_proj_out = buf.o_proj_out->slice(0, 0, seq_len);
  // q
  llaisys::ops::linear(q_out,
                       input,
                       weights_.attn_q_w[layer],
                       weights_.attn_q_b[layer]);
  auto pos_ids = internal_buffers_.pos_ids->slice(0, pid, pid + seq_len);
  q_out = q_out->reshape({seq_len, meta_.nh, meta_.dh});
  llaisys::ops::rope(q_out, q_out, pos_ids, meta_.theta);
  // k
  llaisys::ops::linear(k_cache_out,
                       input,
                       weights_.attn_k_w[layer],
                       weights_.attn_k_b[layer]);
  k_cache_out = k_cache_out->reshape({seq_len, meta_.nkvh, meta_.dh});
  llaisys::ops::rope(k_cache_out, k_cache_out, pos_ids, meta_.theta);
  // v
  llaisys::ops::linear(v_cache_out,
                       input,
                       weights_.attn_v_w[layer],
                       weights_.attn_v_b[layer]);
  
  // attention
  k_out = k_out->reshape({pid + seq_len, meta_.nkvh, meta_.dh});
  v_out = v_out->reshape({pid + seq_len, meta_.nkvh, meta_.dh});
  float scale = 1.0f / sqrtf(static_cast<float>(meta_.dh));
  llaisys::ops::self_attention(attn_out,
                               internal_buffers_.s, // temp buffer, not need to slice
                               q_out,
                               k_out,
                               v_out,
                               scale);
  // output projection
  llaisys::ops::linear(o_proj_out,
                       attn_out,
                       weights_.attn_o_w[layer],
                       nullptr);
  return o_proj_out;
}

tensor_t Qwen2Model::forwardFeedForward(tensor_t input, size_t layer) {
  auto& buf = internal_buffers_;
  size_t seq_len = input->shape()[0];
  auto mlp_gate_out = buf.mlp_gate_out->slice(0, 0, seq_len);
  auto mlp_up_out =  buf.mlp_up_out->slice(0, 0, seq_len);
  auto mlp_down_out = buf.mlp_down_out->slice(0, 0, seq_len);
  // mlp gate
  llaisys::ops::linear(mlp_gate_out,
                       input,
                       weights_.mlp_gate_w[layer],
                       nullptr);
  // mlp up
  llaisys::ops::linear(mlp_up_out,
                       input,
                       weights_.mlp_up_w[layer],
                       nullptr);
  // swiglu
  llaisys::ops::swiglu(mlp_gate_out,
                       mlp_gate_out,
                       mlp_up_out);
  // mlp down
  llaisys::ops::linear(mlp_down_out,
                       mlp_gate_out,
                       weights_.mlp_down_w[layer],
                       nullptr);
  
  return mlp_down_out;
}

tensor_t Qwen2Model::forwardLMHead(tensor_t input) {
  auto& buf = internal_buffers_;
  size_t seq_len = input->shape()[0];
  auto lm_head_out = buf.lm_head_out->slice(0, 0, seq_len);
  llaisys::ops::linear(lm_head_out,
                       input,
                       weights_.out_embed,
                       nullptr);
  return lm_head_out;
}

tensor_t Qwen2Model::forwardEncoder(tensor_t input, size_t layer) {
  auto pre_atten_rms_out = forwardRMSNorm(input,
                                          weights_.attn_norm_w[layer],
                                          meta_.epsilon);
  auto atten_out = forwardAttention(pre_atten_rms_out, layer);
  // residual
  llaisys::ops::add(atten_out, input, atten_out);
  auto post_atten_rms_out = forwardRMSNorm(atten_out,
                                           weights_.mlp_norm_w[layer],
                                           meta_.epsilon);
  auto ffn_out = forwardFeedForward(post_atten_rms_out, layer);
  // residual
  llaisys::ops::add(ffn_out, atten_out, ffn_out);
  return ffn_out;
}


tensor_t Qwen2Model::forward(tensor_t input) {
  auto hidden = forwardEmbedding(input);
  // encoder layers
  for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
    // printf("Forwarding layer %zu/%zu\n", layer + 1, meta_.nlayer);
    hidden = forwardEncoder(hidden, layer);
  }
  // output rms norm
  auto lm_norm_out = forwardRMSNorm(hidden,
                                    weights_.out_norm_w,
                                    meta_.epsilon);
  // lm head
  auto lm_head_out = forwardLMHead(lm_norm_out);
    
  // get argmax token id (only for the last token position)
  size_t seq_len = hidden->shape()[0];
  auto last_logits = lm_head_out->slice(0, seq_len - 1, seq_len);  // [1, vocab_size]
  auto token_id = internal_buffers_.token_ids->slice(0, 0, 1);
  llaisys::ops::argmax(token_id, internal_buffers_.max_logits, last_logits);

  return token_id;
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t ntoken) {
  // if(is_decoding) *reinterpret_cast<int64_t*>(internal_buffers_.pos_id->data()) = pos_id;
  // printf("Infer called with ntoken=%zu, pos_id=%ld\n", ntoken, pos_id);
  auto token_ids_tensor = internal_buffers_.token_ids->slice(0, 0, ntoken);
  token_ids_tensor->load(token_ids);
  auto next_token_id_tensor = forward(token_ids_tensor);
  int64_t next_token_id = *reinterpret_cast<int64_t*>(
      next_token_id_tensor->data());
  pos_id += (int64_t)ntoken;
  return next_token_id;
}

} // namespace llaisys