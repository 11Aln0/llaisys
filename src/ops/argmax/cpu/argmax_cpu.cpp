
#include "../../../utils.hpp"
#include "argmax_cpu.hpp"

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    T max_value = vals[0];
    int64_t max_index = 0;
    for (size_t i = 1; i < numel; i++) {
        if (vals[i] > max_value) {
            max_value = vals[i];
            max_index = static_cast<int64_t>(i);
        }
    }
    *max_val = max_value;
    *max_idx = max_index;
}

namespace llaisys::ops::cpu {

void argmax(std::byte *max_idx_, std::byte *max_val_, const std::byte *vals_, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx_), reinterpret_cast<float *>(max_val_),
                        reinterpret_cast<const float *>(vals_), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx_), reinterpret_cast<llaisys::bf16_t *>(max_val_),
                        reinterpret_cast<const llaisys::bf16_t *>(vals_), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx_), reinterpret_cast<llaisys::fp16_t *>(max_val_),
                        reinterpret_cast<const llaisys::fp16_t *>(vals_), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
  }
}