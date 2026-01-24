#include "op.hpp"
#include "cpu/rope_cpu.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    size_t seq_len = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t head_dim = in->shape()[2];

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return llaisys::ops::cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                                      out->dtype(), seq_len, nhead, head_dim);
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
