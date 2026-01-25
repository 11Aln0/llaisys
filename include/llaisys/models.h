#ifndef LLAISYS_MODELS_H
#define LLAISYS_MODELS_H

#include "tensor.h"

__C {
    // ============ Qwen2 Model ============

    // Opaque handle type
    typedef struct LlaisysQwen2Model *llaisysQwen2Model_t;

    // Meta structure (public, for initialization)
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    // Weights structure (public, for weight loading)
    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    __export llaisysQwen2Model_t llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice);

    __export void llaisysQwen2ModelDestroy(
        llaisysQwen2Model_t model);

    __export LlaisysQwen2Weights *llaisysQwen2ModelWeights(
        llaisysQwen2Model_t model);

    __export int64_t llaisysQwen2ModelInfer(
        llaisysQwen2Model_t model,
        int64_t *token_ids,
        size_t ntoken);
}

#endif // LLAISYS_MODELS_H