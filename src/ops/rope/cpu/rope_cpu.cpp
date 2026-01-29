
#include "../../../utils.hpp"
#include "rope_cpu.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
void _rope(T *out, const T *in, const int64_t *pos_id, float theta,
               size_t seq_len, size_t nhead, size_t head_dim) {
    using llaisys::utils::cast;
    // head_dim must be even
    int64_t nh = (int64_t)nhead;
    int64_t hd = (int64_t)head_dim;

    const int64_t half_dim = hd / 2;

    int64_t parallel_n = (int64_t)seq_len * nh;

    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < parallel_n; ++idx) {
        int64_t s = idx / nh;
        int64_t h = idx % nh;

        // position id
        float pos = static_cast<float>(pos_id[s]);
        const int64_t base = (s * nh + h) * hd;

        for (int64_t i = 0; i < half_dim; ++i) {
            // compute rotation angle
            float freq = std::pow(theta, 2.0f * (float)i / (float)hd);
            
            float angle = pos / freq;

            float cos_t = std::cos(angle);
            float sin_t = std::sin(angle);

            int64_t even_idx = base + i;
            int64_t odd_idx  = base + half_dim + i;

            float x_even = cast<float>(in[even_idx]);
            float x_odd  = cast<float>(in[odd_idx]);

            float y_even = x_even * cos_t - x_odd * sin_t;
            float y_odd  = x_odd * cos_t + x_even * sin_t;

            out[even_idx] = cast<T>(y_even);
            out[odd_idx]  = cast<T>(y_odd);
        }
    }
    
}

namespace llaisys::ops::cpu {

void rope(std::byte *out_, const std::byte *in_, const std::byte *ops_id_, float theta,
            llaisysDataType_t dtype, size_t seq_len, size_t nhead, size_t head_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return _rope(reinterpret_cast<float *>(out_),
                     reinterpret_cast<const float *>(in_),
                     reinterpret_cast<const int64_t *>(ops_id_),
                     theta,
                     seq_len,
                     nhead,
                     head_dim);
    case LLAISYS_DTYPE_BF16:
        return _rope(reinterpret_cast<llaisys::bf16_t *>(out_),
                     reinterpret_cast<const llaisys::bf16_t *>(in_),
                     reinterpret_cast<const int64_t *>(ops_id_),
                     theta,
                     seq_len,
                     nhead,
                     head_dim);
    case LLAISYS_DTYPE_F16:
        return _rope(reinterpret_cast<llaisys::fp16_t *>(out_),
                     reinterpret_cast<const llaisys::fp16_t *>(in_),
                     reinterpret_cast<const int64_t *>(ops_id_),
                     theta,
                     seq_len,
                     nhead,
                     head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}