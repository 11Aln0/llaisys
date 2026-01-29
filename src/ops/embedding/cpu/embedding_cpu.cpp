
#include "../../../utils.hpp"
#include "embedding_cpu.hpp"

template <typename T>
void _embedding(T *out, const int64_t *index, const T *weight,
               size_t index_numel,
               size_t embedding_dim) {
    for (size_t i = 0; i < index_numel; i++) {
        uint64_t idx = (uint64_t)index[i];
        for (size_t j = 0; j < embedding_dim; j++) {
            out[i * embedding_dim + j] = weight[idx * embedding_dim + j];
        }
    }
}

namespace llaisys::ops::cpu {

void embedding(std::byte *out_, const std::byte *index_, const std::byte *weight_,
               llaisysDataType_t dtype,
               size_t index_numel,
               size_t embedding_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return _embedding(reinterpret_cast<float *>(out_),
                         reinterpret_cast<const int64_t *>(index_),
                         reinterpret_cast<const float *>(weight_),
                         index_numel,
                         embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return _embedding(reinterpret_cast<llaisys::bf16_t *>(out_),
                         reinterpret_cast<const int64_t *>(index_),
                         reinterpret_cast<const llaisys::bf16_t *>(weight_),
                         index_numel,
                         embedding_dim);
    case LLAISYS_DTYPE_F16:
        return _embedding(reinterpret_cast<llaisys::fp16_t *>(out_),
                         reinterpret_cast<const int64_t *>(index_),
                         reinterpret_cast<const llaisys::fp16_t *>(weight_),
                         index_numel,
                         embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}