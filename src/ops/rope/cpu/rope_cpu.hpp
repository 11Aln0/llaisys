#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out_, const std::byte *in_, const std::byte *ops_id_, float theta,
            llaisysDataType_t dtype, size_t seq_len, size_t nhead, size_t head_dim);
}