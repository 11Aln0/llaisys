#pragma once

#include <cstddef>
#include "types.hpp"

namespace llaisys {

template <typename T>
typename std::enable_if<std::is_same<T, llaisys::bf16_t>::value || std::is_same<T, llaisys::fp16_t>::value, T>::type
operator+(const T &a, const T &b) {
    return utils::cast<T>(utils::cast<float>(a) + utils::cast<float>(b));
}

template <typename T>
typename std::enable_if<std::is_same<T, llaisys::bf16_t>::value || std::is_same<T, llaisys::fp16_t>::value, bool>::type
operator>(const T &a, const T &b) {
    return utils::cast<float>(a) > utils::cast<float>(b);
}

template <typename T>
typename std::enable_if<std::is_same<T, llaisys::bf16_t>::value || std::is_same<T, llaisys::fp16_t>::value, float>::type
operator*(const T &a, const T &b) {
    return utils::cast<float>(a) * utils::cast<float>(b);
}

}