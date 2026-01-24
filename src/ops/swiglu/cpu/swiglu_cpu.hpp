#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void swiglu(std::byte *out_, const std::byte *gate_, const std::byte *up_,
            llaisysDataType_t dtype, size_t size);
}