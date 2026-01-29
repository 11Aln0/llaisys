#include <cstddef>
#include <omp.h>

#include "../../utils.hpp"
#include "gemm.h"

template <typename T>
T defaultEpilogue(float acc, int64_t batch_idx, int64_t row_idx, int64_t col_idx) {
    return llaisys::utils::cast<T>(acc);
}

template <
    typename T,
    Layout LayoutB,
    typename Epilogue = decltype(defaultEpilogue<T>)
>
void batch_gemm_cpu_blocked_omp(
    const T* A,
    const T* B,
    T* C,
    size_t batch,
    size_t M,
    size_t N,
    size_t K,
    size_t batch_stride_A,
    size_t batch_stride_B,
    size_t batch_stride_C,
    size_t lda,
    size_t ldb,
    size_t ldc,
    Epilogue epilogue = defaultEpilogue<T>
) {
    using llaisys::utils::cast;

    omp_set_nested(1);
#pragma omp parallel for schedule(static)
  for (int64_t ib = 0; ib < (int64_t)batch; ++ib) {
    const T* A_batch = A + ib * batch_stride_A;
    const T* B_batch = B + ib * batch_stride_B;
    T*       C_batch = C + ib * batch_stride_C;
    gemm_cpu_blocked_omp<T, LayoutB>(A_batch, B_batch, C_batch,
                         M, N, K,
                         lda, ldb, ldc,
                         [epilogue, ib](float acc, int64_t im, int64_t in) {
                             return epilogue(acc, ib, im, in);
                         });
  }
}