#pragma once

#include "llaisys/tensor.h"
#include "../tensor/tensor.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace llaisys {

// Internal Qwen2 Meta structure
struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

// Internal Qwen2 Weights structure
struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;   // a.k.a. model.norm.weight
    tensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
    tensor_t *attn_q_w;
    tensor_t *attn_q_b;
    tensor_t *attn_k_w;
    tensor_t *attn_k_b;
    tensor_t *attn_v_w;
    tensor_t *attn_v_b;
    tensor_t *attn_o_w;
    tensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
    tensor_t *mlp_gate_w;
    tensor_t *mlp_up_w;
    tensor_t *mlp_down_w;
};

struct Qwen2InternalBuffers {
  tensor_t pos_id; // [max_seq_len]
  tensor_t embed_out; // [max_seq_len, hidden_size]
  tensor_t rms_norm_out; // for decoder rmsx2 + lm_head rms [max_seq_len, hidden_size]
  // attention
  tensor_t q_out; // [max_seq_len, num_heads * head_dim]
  tensor_t s; // buffer for attention_score [num_heads, max_seq_len, max_seq_len]
  tensor_t attn_out; // [max_seq_len, num_heads * head_dim]
  tensor_t o_proj_out; // [max_seq_len, hidden_size]
  // no need for redisual
  // feed-forward
  tensor_t mlp_gate_out; // [max_seq_len, inter_size] (reused for swiglu out) 
  tensor_t mlp_up_out; // [max_seq_len, inter_size]
  tensor_t mlp_down_out; // [max_seq_len, hidden_size]

  tensor_t lm_head_out; // [max_seq_len, vocab_size]
};

struct Qwen2KVCache {
  tensor_t* k; // [max_seq_len, num_kv_heads * head_dim] x nlayer
  tensor_t* v; // [max_seq_len, num_kv_heads * head_dim] x nlayer
};

class Qwen2Model {
public:
  Qwen2Model(const Qwen2Meta& meta, llaisysDeviceType_t device, 
              const std::vector<int>& device_ids);
  ~Qwen2Model();

  Qwen2Weights* weights() { return &weights_; }
  int64_t infer(const int64_t* token_ids, size_t ntoken);

private:
  // mem
  void initEncoderLayerWeight(int layer);
  void initEncoderLayersWeight();
  void initWeights();
  void initInternalBuffers();
  void initKVCache();
  // forward
  void forwardEmbedding(const tensor_t input_ids);
  void forwardRMSNorm(const tensor_t input, const tensor_t weight,
                      tensor_t output, float epsilon);
  void forwardAttention(tensor_t input, int layer);
  void forwardFeedForward(int layer);
  void forwardEncoder();
  void forwardLMHead();

private:
  bool is_decode_stage = false;
  Qwen2Meta meta_;
  Qwen2Weights weights_;
  Qwen2InternalBuffers internal_buffers_;
  Qwen2KVCache kvcache_;
  llaisysDeviceType_t device_;
  std::vector<int> device_ids_;
};

using qwen2_model_t = std::shared_ptr<Qwen2Model>;

} // namespace llaisys
