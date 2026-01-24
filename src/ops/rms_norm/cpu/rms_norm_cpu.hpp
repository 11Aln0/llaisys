#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out_, const std::byte *in_, const std::byte *weight_, float eps,
            llaisysDataType_t dtype, size_t batch_size, size_t feature_dim);
}