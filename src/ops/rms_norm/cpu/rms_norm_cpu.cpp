
#include "../../../utils.hpp"
#include "rms_norm_cpu.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
void _rms_norm(T *out, const T *in, const T *weight, float eps,
               size_t batch_size, size_t feature_dim) {
    int64_t fd = (int64_t)feature_dim;
    int64_t bs = (int64_t)batch_size;

    using llaisys::utils::cast;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < bs; ++i) {
        float sum_of_squares = 0.0f;
        for (int64_t j = 0; j < fd; ++j) {
            sum_of_squares += in[i * fd + j] * in[i * fd + j];
        }
        float rms = std::sqrt(sum_of_squares / (float)fd + eps);

        for (int64_t j = 0; j < fd; ++j) {
            out[i * fd+ j] = cast<T>(cast<float>(in[i * fd + j]) / rms * cast<float>(weight[j]));  
        }
    }
}

namespace llaisys::ops::cpu {

void rms_norm(std::byte *out_, const std::byte *in_, const std::byte *weight_, float eps,
            llaisysDataType_t dtype, size_t batch_size, size_t feature_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return _rms_norm(reinterpret_cast<float *>(out_),
                         reinterpret_cast<const float *>(in_),
                         reinterpret_cast<const float *>(weight_),
                         eps,
                         batch_size,
                         feature_dim);
    case LLAISYS_DTYPE_BF16:
        return _rms_norm(reinterpret_cast<llaisys::bf16_t *>(out_),
                         reinterpret_cast<const llaisys::bf16_t *>(in_),
                         reinterpret_cast<const llaisys::bf16_t *>(weight_),
                         eps,
                         batch_size,
                         feature_dim);
    case LLAISYS_DTYPE_F16:
        return _rms_norm(reinterpret_cast<llaisys::fp16_t *>(out_),
                         reinterpret_cast<const llaisys::fp16_t *>(in_),
                         reinterpret_cast<const llaisys::fp16_t *>(weight_),
                         eps,
                         batch_size,
                         feature_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}