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



};

class Qwen2Model {
public:
    Qwen2Model(const Qwen2Meta& meta, llaisysDeviceType_t device, 
               const std::vector<int>& device_ids);
    ~Qwen2Model();

    Qwen2Weights* weights() { return &weights_; }
    int64_t infer(const int64_t* token_ids, size_t ntoken);

private:
    void initEncoderLayerWeight(int layer);
    void initEncoderLayersWeight();
    void initWeights();
    void initInternalBuffers();
private:
    Qwen2Meta meta_;
    Qwen2Weights weights_;
    llaisysDeviceType_t device_;
    std::vector<int> device_ids_;
};

using qwen2_model_t = std::shared_ptr<Qwen2Model>;

} // namespace llaisys
