#include "../../../utils.hpp"
#include "self_attention_cpu.hpp"
#include <cmath>
#include <omp.h>
#include "../../matmul/batched_gemm.h"
#include "../../matmul/gemm.h"

template <typename T>
void computeScore(T *attn_weight, const T *q, const T *k, float scale,
                  size_t q_len, size_t kv_len, 
                  size_t nhead, size_t n_kvhead, 
                  size_t head_dim) {
  using llaisys::utils::cast;

  const size_t offset = static_cast<size_t>(kv_len - q_len);

  #pragma omp parallel for schedule(static)
  for(size_t ih = 0; ih < nhead; ++ih) {
      const T* q_head = q + ih * head_dim; // [q_len, nhead, head_dim]
      const T* k_head = k + (ih / (nhead / n_kvhead)) * head_dim;
      T*       attn_head = attn_weight + ih * q_len * kv_len;

      gemm_cpu_blocked_omp<T, Layout::ColMajor>(
          q_head, k_head, attn_head,
          q_len,       // M
          kv_len,      // N
          head_dim,    // K
          nhead * head_dim,    // lda
          n_kvhead * head_dim,    // ldb
          kv_len,       // ldc
          [scale, offset](float acc, size_t row, size_t col) {
            // causal mask: col <= row + offset (where offset = kv_len - q_len)
            acc += (col <= row + offset ? 0.0f : -INFINITY);
            return llaisys::utils::cast<T>(acc * scale);
          }
      );
  }
}

template <typename T>
void softmax(T *attn_score, size_t q_len, size_t kv_len, size_t nhead) {
    using llaisys::utils::cast;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t h = 0; h < nhead; ++h) {
        for (size_t q_idx = 0; q_idx < q_len; ++q_idx) {
            const size_t base = (h * q_len + q_idx) * kv_len;

            // find max for numerical stability
            float max_val = -INFINITY;
            for (size_t k_idx = 0; k_idx < kv_len; ++k_idx) {
                float val = cast<float>(attn_score[base + k_idx]);
                max_val = std::max(max_val, val);
            }

            // compute sum of exp
            float sum_exp = 0.0f;
            for (size_t k_idx = 0; k_idx < kv_len; ++k_idx) {
                float val = cast<float>(attn_score[base + k_idx]);
                sum_exp += std::exp(val - max_val);
            }

            // normalize
            for (size_t k_idx = 0; k_idx < kv_len; ++k_idx) {
                float val = cast<float>(attn_score[base + k_idx]);
                attn_score[base + k_idx] = cast<T>(std::exp(val - max_val) / sum_exp);
            }
        }
    }
}

template <typename T>
void computeAttnVal(T *out, const T *attn_weight, const T *v,
                    size_t q_len, size_t kv_len, 
                    size_t nhead, size_t n_kvhead, 
                    size_t head_dim) {
    using llaisys::utils::cast;

    #pragma omp parallel for schedule(static)
    for(size_t ih = 0; ih < nhead; ++ih) {
        const T* attn_head = attn_weight + ih * q_len * kv_len;
        const T* v_head = v + (ih / (nhead / n_kvhead)) * head_dim;  // GQA: map Q head to KV head
        T*       out_head = out + ih * head_dim;

        gemm_cpu_blocked_omp<T, Layout::RowMajor>(
            attn_head, v_head, out_head,
            q_len,       // M
            head_dim,    // N
            kv_len,      // K
            kv_len,      // lda
            n_kvhead * head_dim,    // ldb
            nhead * head_dim        // ldc
        );
    }
}

template <typename T>
void _self_attention(T *out, T *attn_weight, const T *q, const T *k, const T *v, float scale,
                     size_t q_len, size_t kv_len, size_t nhead, size_t n_kvhead, size_t head_dim) {
  omp_set_nested(1);
  computeScore<T>(attn_weight, q, k, scale, q_len, kv_len, nhead, n_kvhead, head_dim);
  softmax<T>(attn_weight, q_len, kv_len, nhead);
  computeAttnVal<T>(out, attn_weight, v, q_len, kv_len, nhead, n_kvhead, head_dim);
}

namespace llaisys::ops::cpu {

void self_attention(std::byte *out_, std::byte *attn_weight_,
                    const std::byte *q_, const std::byte *k_, const std::byte *v_,
                    float scale, llaisysDataType_t dtype,
                    size_t q_len, size_t kv_len, size_t nhead, size_t n_kvhead, size_t head_dim) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return _self_attention(reinterpret_cast<float *>(out_),
                               reinterpret_cast<float *>(attn_weight_),
                               reinterpret_cast<const float *>(q_),
                               reinterpret_cast<const float *>(k_),
                               reinterpret_cast<const float *>(v_),
                               scale, q_len, kv_len, nhead, n_kvhead, head_dim);
    case LLAISYS_DTYPE_BF16:
        return _self_attention(reinterpret_cast<llaisys::bf16_t *>(out_),
                               reinterpret_cast<llaisys::bf16_t *>(attn_weight_),
                               reinterpret_cast<const llaisys::bf16_t *>(q_),
                               reinterpret_cast<const llaisys::bf16_t *>(k_),
                               reinterpret_cast<const llaisys::bf16_t *>(v_),
                               scale, q_len, kv_len, nhead, n_kvhead, head_dim);
    case LLAISYS_DTYPE_F16:
        return _self_attention(reinterpret_cast<llaisys::fp16_t *>(out_),
                               reinterpret_cast<llaisys::fp16_t *>(attn_weight_),
                               reinterpret_cast<const llaisys::fp16_t *>(q_),
                               reinterpret_cast<const llaisys::fp16_t *>(k_),
                               reinterpret_cast<const llaisys::fp16_t *>(v_),
                               scale, q_len, kv_len, nhead, n_kvhead, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}
