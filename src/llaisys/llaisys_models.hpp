#pragma once

#include "llaisys/models.h"
#include "llaisys_tensor.hpp"
#include "../models/qwen2.hpp"

__C {
    typedef struct LlaisysQwen2Model {
        llaisys::qwen2_model_t model;
    } LlaisysQwen2Model;
}
