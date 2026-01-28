#include "llaisys_models.hpp"

#include <vector>
#include <string>

// Helper: wrap internal tensor into public handle
static LlaisysTensor *wrapTensor(const llaisys::tensor_t &t) {
    return t ? new LlaisysTensor{t} : nullptr;
}

// Helper: wrap an array of internal tensors into public handles
static llaisysTensor_t *wrapTensorArray(const llaisys::tensor_t *arr, size_t n) {
    if (!arr || n == 0) return nullptr;
    auto *out = new llaisysTensor_t[n];
    for (size_t i = 0; i < n; ++i) {
        out[i] = wrapTensor(arr[i]);
    }
    return out;
}

// Convert internal weights to public weights struct
static LlaisysQwen2Weights *toPublicWeights(const llaisys::Qwen2Weights *w, size_t nlayer) {
    if (!w) return nullptr;
    auto *pub = new LlaisysQwen2Weights{};
    pub->in_embed = wrapTensor(w->in_embed);
    pub->out_embed = wrapTensor(w->out_embed);
    pub->out_norm_w = wrapTensor(w->out_norm_w);

    pub->attn_norm_w = wrapTensorArray(w->attn_norm_w, nlayer);
    pub->attn_q_w = wrapTensorArray(w->attn_q_w, nlayer);
    pub->attn_q_b = wrapTensorArray(w->attn_q_b, nlayer);
    pub->attn_k_w = wrapTensorArray(w->attn_k_w, nlayer);
    pub->attn_k_b = wrapTensorArray(w->attn_k_b, nlayer);
    pub->attn_v_w = wrapTensorArray(w->attn_v_w, nlayer);
    pub->attn_v_b = wrapTensorArray(w->attn_v_b, nlayer);
    pub->attn_o_w = wrapTensorArray(w->attn_o_w, nlayer);

    pub->mlp_norm_w = wrapTensorArray(w->mlp_norm_w, nlayer);
    pub->mlp_gate_w = wrapTensorArray(w->mlp_gate_w, nlayer);
    pub->mlp_up_w = wrapTensorArray(w->mlp_up_w, nlayer);
    pub->mlp_down_w = wrapTensorArray(w->mlp_down_w, nlayer);
    return pub;
}

// Helper: destroy a public weights struct
static void destroyPublicWeights(LlaisysQwen2Weights *w, size_t nlayer) {
    if (!w) return;
    auto destroyHandle = [](llaisysTensor_t t) {
        delete t;
    };
    auto destroyArray = [&](llaisysTensor_t *arr) {
        if (!arr) return;
        for (size_t i = 0; i < nlayer; ++i) {
            destroyHandle(arr[i]);
        }
        delete[] arr;
    };

    destroyHandle(w->in_embed);
    destroyHandle(w->out_embed);
    destroyHandle(w->out_norm_w);

    destroyArray(w->attn_norm_w);
    destroyArray(w->attn_q_w);
    destroyArray(w->attn_q_b);
    destroyArray(w->attn_k_w);
    destroyArray(w->attn_k_b);
    destroyArray(w->attn_v_w);
    destroyArray(w->attn_v_b);
    destroyArray(w->attn_o_w);
    destroyArray(w->mlp_norm_w);
    destroyArray(w->mlp_gate_w);
    destroyArray(w->mlp_up_w);
    destroyArray(w->mlp_down_w);

    delete w;
}

// Helper to convert public meta to internal meta
static llaisys::Qwen2Meta toInternalMeta(const LlaisysQwen2Meta *meta) {
    llaisys::Qwen2Meta internal;
    internal.dtype = meta->dtype;
    internal.nlayer = meta->nlayer;
    internal.hs = meta->hs;
    internal.nh = meta->nh;
    internal.nkvh = meta->nkvh;
    internal.dh = meta->dh;
    internal.di = meta->di;
    internal.maxseq = meta->maxseq;
    internal.voc = meta->voc;
    internal.epsilon = meta->epsilon;
    internal.theta = meta->theta;
    internal.end_token = meta->end_token;
    return internal;
}

__C {
    __export llaisysQwen2Model_t llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                         llaisysDeviceType_t device,
                                                         int *device_ids,
                                                         int ndevice) {
        std::vector<int> device_ids_vec(device_ids, device_ids + ndevice);
        auto internal_meta = toInternalMeta(meta);
        auto model = std::make_shared<llaisys::Qwen2Model>(internal_meta, device, device_ids_vec);
        return new LlaisysQwen2Model{model, nullptr};
    }

    __export void llaisysQwen2ModelDestroy(llaisysQwen2Model_t model) {
        if (model->public_weights) {
            destroyPublicWeights(model->public_weights, model->model->numLayers());
            model->public_weights = nullptr;
        }
        delete model;
    }

    __export LlaisysQwen2Weights *llaisysQwen2ModelWeights(llaisysQwen2Model_t model) {
        if (!model->public_weights) {
            model->public_weights = toPublicWeights(model->model->weights(), model->model->numLayers());
        }
        return model->public_weights;
    }

    __export int64_t llaisysQwen2ModelInfer(
        llaisysQwen2Model_t model,
        int64_t *token_ids,
        size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }
}
