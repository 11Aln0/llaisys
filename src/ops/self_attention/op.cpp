#include "op.hpp"
#include "cpu/self_attention_cpu.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t attn_weight, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, attn_weight, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), attn_weight->dtype(), q->dtype(), k->dtype(), v->dtype());

    size_t q_len = q->shape()[0];
    size_t kv_len = k->shape()[0];
    size_t nhead = q->shape()[1];
    size_t n_kvhead = k->shape()[1];
    size_t head_dim = q->shape()[2];

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return llaisys::ops::cpu::self_attention(attn_val->data(), attn_weight->data(),
                                                 q->data(), k->data(), v->data(),
                                                 scale, attn_val->dtype(), q_len, kv_len, nhead, n_kvhead, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
