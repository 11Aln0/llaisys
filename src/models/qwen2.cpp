#include "qwen2.hpp"
#include "../tensor/tensor.hpp"

namespace llaisys {

Qwen2Model::Qwen2Model(const Qwen2Meta& meta, llaisysDeviceType_t device,
                       const std::vector<int>& device_ids)
    : meta_(meta), device_(device), device_ids_(device_ids) {
    
    initWeights();
}

Qwen2Model::~Qwen2Model() {
    // Cleanup layer weight arrays
    delete[] weights_.attn_norm_w;
    delete[] weights_.attn_q_w;
    delete[] weights_.attn_q_b;
    delete[] weights_.attn_k_w;
    delete[] weights_.attn_k_b;
    delete[] weights_.attn_v_w;
    delete[] weights_.attn_v_b;
    delete[] weights_.attn_o_w;
    delete[] weights_.mlp_norm_w;
    delete[] weights_.mlp_gate_w;
    delete[] weights_.mlp_up_w;
    delete[] weights_.mlp_down_w;
    weights_.in_embed = nullptr;
    weights_.out_embed = nullptr;
    weights_.out_norm_w = nullptr;
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t ntoken) {
    // TODO: Implement inference
    return -1;
}

void Qwen2Model::initEncoderLayerWeight(int layer) {
  using llaisys::Tensor;
  auto& w = weights_;
  auto dtype = meta_.dtype;
  auto devId = device_ids_[0];
  // input_layernorm.weight
  w.attn_norm_w[layer] = Tensor::create({meta_.hs}, dtype, device_, devId);
  // qkv
  size_t q_hdim = meta_.nh * meta_.dh;
  size_t kv_hdim = meta_.nkvh * meta_.dh;
  w.attn_q_w[layer] = Tensor::create({q_hdim, meta_.hs}, dtype, device_, devId);
  w.attn_q_b[layer] = Tensor::create({q_hdim}, dtype, device_, devId);
  w.attn_k_w[layer] = Tensor::create({kv_hdim, meta_.hs}, dtype, device_, devId);
  w.attn_k_b[layer] = Tensor::create({kv_hdim}, dtype, device_, devId);
  w.attn_v_w[layer] = Tensor::create({kv_hdim, meta_.hs}, dtype, device_, devId); 
  w.attn_v_b[layer] = Tensor::create({kv_hdim}, dtype, device_, devId);
  // output projection
  w.attn_o_w[layer] = Tensor::create({meta_.hs, q_hdim}, dtype, device_, devId);
  // post_attention_layernorm.weight
  w.mlp_norm_w[layer] = Tensor::create({meta_.hs}, dtype, device_, devId);
  // mlp
  w.mlp_gate_w[layer] = Tensor::create({meta_.di, meta_.hs}, dtype, device_, devId);
  w.mlp_up_w[layer] = Tensor::create({meta_.di, meta_.hs}, dtype, device_, devId);
  w.mlp_down_w[layer] = Tensor::create({meta_.hs, meta_.di}, dtype, device_, devId);
}

void Qwen2Model::initEncoderLayersWeight() {
  for (size_t i = 0; i < meta_.nlayer; ++i) {
    initEncoderLayerWeight(i);
  }
}

void Qwen2Model::initWeights() {
  llaisysDataType_t dtype = meta_.dtype;
  size_t nlayer = meta_.nlayer;
  weights_ = {
      .in_embed = nullptr,
      .out_embed = nullptr,
      .out_norm_w = nullptr,
      .attn_norm_w = new tensor_t[nlayer],
      .attn_q_w = new tensor_t[nlayer],
      .attn_q_b = new tensor_t[nlayer],
      .attn_k_w = new tensor_t[nlayer],
      .attn_k_b = new tensor_t[nlayer],
      .attn_v_w = new tensor_t[nlayer],
      .attn_v_b = new tensor_t[nlayer],
      .attn_o_w = new tensor_t[nlayer],
      .mlp_norm_w = new tensor_t[nlayer],
      .mlp_gate_w = new tensor_t[nlayer],
      .mlp_up_w = new tensor_t[nlayer],
      .mlp_down_w = new tensor_t[nlayer],
  };

  auto embed_weight = llaisys::Tensor::create({meta_.voc, meta_.hs}, dtype, device_, device_ids_[0]);
  weights_.in_embed = embed_weight;
  weights_.out_embed = embed_weight;
  weights_.out_norm_w = llaisys::Tensor::create({meta_.hs}, dtype, device_, device_ids_[0]);
  
  initEncoderLayersWeight();
}

void Qwen2Model::initInternalBuffers() {


}

} // namespace llaisys
