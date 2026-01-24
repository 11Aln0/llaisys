#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out_, const std::byte *in_, const std::byte *weight_, const std::byte *bias_,
            llaisysDataType_t dtype, size_t m, size_t n, size_t k);
}