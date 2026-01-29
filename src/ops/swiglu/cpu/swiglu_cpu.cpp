
#include "../../../utils.hpp"
#include "swiglu_cpu.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
void _swiglu(T *out, const T *gate, const T *up, size_t size) {
    using llaisys::utils::cast;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        // SwiGLU: swish(gate) * up = (gate * sigmoid(gate)) * up
        float g = cast<float>(gate[i]);
        float u = cast<float>(up[i]);

        // sigmoid(g) = 1 / (1 + exp(-g))
        float sigmoid_g = 1.0f / (1.0f + std::exp(-g));

        // swish(g) = g * sigmoid(g)
        float swish_g = g * sigmoid_g;

        // result = swish(gate) * up
        float result = swish_g * u;

        out[i] = cast<T>(result);
    }
}

namespace llaisys::ops::cpu {

void swiglu(std::byte *out_, const std::byte *gate_, const std::byte *up_,
            llaisysDataType_t dtype, size_t size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return _swiglu(reinterpret_cast<float *>(out_),
                       reinterpret_cast<const float *>(gate_),
                       reinterpret_cast<const float *>(up_),
                       size);
    case LLAISYS_DTYPE_BF16:
        return _swiglu(reinterpret_cast<llaisys::bf16_t *>(out_),
                       reinterpret_cast<const llaisys::bf16_t *>(gate_),
                       reinterpret_cast<const llaisys::bf16_t *>(up_),
                       size);
    case LLAISYS_DTYPE_F16:
        return _swiglu(reinterpret_cast<llaisys::fp16_t *>(out_),
                       reinterpret_cast<const llaisys::fp16_t *>(gate_),
                       reinterpret_cast<const llaisys::fp16_t *>(up_),
                       size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}