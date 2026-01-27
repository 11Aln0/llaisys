#include "llaisys_models.hpp"

#include <vector>
#include <string>

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
        return new LlaisysQwen2Model{model};
    }

    __export void llaisysQwen2ModelDestroy(llaisysQwen2Model_t model) {
        delete model;
    }

    __export LlaisysQwen2Weights *llaisysQwen2ModelWeights(llaisysQwen2Model_t model) {
        // Note: This returns internal weights casted to public type
        // They have the same memory layout
        return reinterpret_cast<LlaisysQwen2Weights*>(model->model->weights());
    }

    __export int64_t llaisysQwen2ModelInfer(
        llaisysQwen2Model_t model,
        int64_t *token_ids,
        size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }
}
