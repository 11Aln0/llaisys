#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *out_, std::byte *attn_weight_, 
                    const std::byte *q_, const std::byte *k_, const std::byte *v_,
                    float scale, llaisysDataType_t dtype,
                    size_t q_len, size_t kv_len, size_t nhead, size_t n_kvhead, size_t head_dim);
}
