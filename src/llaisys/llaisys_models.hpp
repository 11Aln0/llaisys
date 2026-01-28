#pragma once

#include "llaisys/models.h"
#include "llaisys_tensor.hpp"
#include "../models/qwen2.hpp"

__C {
    typedef struct LlaisysQwen2Model {
        llaisys::qwen2_model_t model;
        // Cached public weights handle, lazily materialized from internal weights
        LlaisysQwen2Weights *public_weights = nullptr;
    } LlaisysQwen2Model;
}
