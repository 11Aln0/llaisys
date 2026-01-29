
#include "../../../utils.hpp"
#include "rms_norm_cpu.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
void _rms_norm(T *out, const T *in, const T *weight, float eps,
               size_t batch_size, size_t feature_dim) {
    using llaisys::utils::cast;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)batch_size; ++i) {
        float sum_of_squares = 0.0f;
        for (int64_t j = 0; j < (int64_t)feature_dim; ++j) {
            sum_of_squares += in[i * feature_dim + j] * in[i * feature_dim + j];
        }
        float rms = std::sqrt(sum_of_squares / (float)feature_dim + eps);  

        for (int64_t j = 0; j < (int64_t)feature_dim; ++j) {
            out[i * feature_dim + j] = cast<T>(cast<float>(in[i * feature_dim + j]) / rms * cast<float>(weight[j]));  
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