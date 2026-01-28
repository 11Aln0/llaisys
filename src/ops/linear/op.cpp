#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    int m = in->shape()[0], n = weight->shape()[0], k = weight->shape()[1];

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return llaisys::ops::cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr,
                                         out->dtype(), m, n, k);
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
