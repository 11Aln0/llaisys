
#include "../../../utils.hpp"
#include "linear_cpu.hpp"
#include "../../matmul/gemm.h"

template <typename T>
void _linear(T *out, const T *in, const T *weight, const T* bias,
             size_t m, size_t n, size_t k) {
    gemm_cpu_blocked_omp<T, Layout::ColMajor>(
        in, weight, out,
        m,        // M
        n,        // N
        k,        // K
        k,        // lda
        k,        // ldb
        n,        // ldc
        [bias] (float acc, int64_t im, int64_t in) -> T{
            (void)im;
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[in]);
            }
            return llaisys::utils::cast<T>(acc);
        }
    );
}

namespace llaisys::ops::cpu {

void linear(std::byte *out_, const std::byte *in_, const std::byte *weight_, const std::byte *bias_,
            llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return _linear(reinterpret_cast<float *>(out_),
                       reinterpret_cast<const float *>(in_),
                       reinterpret_cast<const float *>(weight_),
                       reinterpret_cast<const float *>(bias_),
                       m, n, k);
    case LLAISYS_DTYPE_BF16:
        return _linear(reinterpret_cast<llaisys::bf16_t *>(out_),
                       reinterpret_cast<const llaisys::bf16_t *>(in_),
                       reinterpret_cast<const llaisys::bf16_t *>(weight_),
                       reinterpret_cast<const llaisys::bf16_t *>(bias_),
                       m, n, k);
    case LLAISYS_DTYPE_F16:
        return _linear(reinterpret_cast<llaisys::fp16_t *>(out_),
                       reinterpret_cast<const llaisys::fp16_t *>(in_),
                       reinterpret_cast<const llaisys::fp16_t *>(weight_),
                       reinterpret_cast<const llaisys::fp16_t *>(bias_),
                       m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}