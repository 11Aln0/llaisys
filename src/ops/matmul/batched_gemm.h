#include <cstddef>
#include <omp.h>

#include "../../utils.hpp"
#include "gemm.h"

template <typename T>
T defaultEpilogue(float acc, int batch_idx, int row_idx, int col_idx) {
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
    int batch,
    int M,
    int N,
    int K,
    ptrdiff_t batch_stride_A,
    ptrdiff_t batch_stride_B,
    ptrdiff_t batch_stride_C,
    int lda,
    int ldb,
    int ldc,
    Epilogue epilogue = defaultEpilogue<T>
) {
    using llaisys::utils::cast;

    omp_set_nested(1);
#pragma omp parallel for schedule(static)
  for (int ib = 0; ib < batch; ++ib) {
    const T* A_batch = A + ib * batch_stride_A;
    const T* B_batch = B + ib * batch_stride_B;
    T*       C_batch = C + ib * batch_stride_C;
    gemm_cpu_blocked_omp<T, LayoutB>(A_batch, B_batch, C_batch,
                         M, N, K,
                         lda, ldb, ldc,
                         [epilogue, ib](float acc, int im, int in) {
                             return epilogue(acc, ib, im, in);
                         });
  }
}