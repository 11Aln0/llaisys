#pragma once
#include <cstddef>
#include <omp.h>

#include "../../utils.hpp"

enum class Layout {
    RowMajor,
    ColMajor
};

template <typename T>
T defaultGEMMEpilogue(float acc, int im, int in) {
    return llaisys::utils::cast<T>(acc);
}

template <
    typename T,
    Layout LayoutB,
    typename Epilogue = decltype(defaultGEMMEpilogue<T>)
>
void gemm_cpu_blocked_omp(
    const T* A,
    const T* B,
    T* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    Epilogue epilogue = defaultGEMMEpilogue<T>
) {
    using llaisys::utils::cast;

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;

#pragma omp parallel for collapse(2) schedule(static)
  for (int i0 = 0; i0 < M; i0 += BM) {
    for (int j0 = 0; j0 < N; j0 += BN) {

      int imax = std::min(i0 + BM, M);
      int jmax = std::min(j0 + BN, N);

      // tile-local float accumulator
      float acc_buf[BM][BN];

      // init
      for (int ii = 0; ii < imax - i0; ++ii) {
          for (int jj = 0; jj < jmax - j0; ++jj) {
              acc_buf[ii][jj] = 0.0f;
          }
      }

      for (int k0 = 0; k0 < K; k0 += BK) {
          int kmax = std::min(k0 + BK, K);

          for (int im = i0; im < imax; ++im) {
              const T* aptr = A + im * lda + k0;
              float*   acc_row = acc_buf[im - i0];

              for (int k = k0; k < kmax; ++k) {
                  float a = cast<float>(aptr[k - k0]);

                  if constexpr (LayoutB == Layout::RowMajor) {
                      const T* bptr = B + k * ldb + j0;
                      for (int in = j0; in < jmax; ++in) {
                          acc_row[in - j0] +=
                              a * cast<float>(bptr[in - j0]);
                      }
                  } else {
                      for (int in = j0; in < jmax; ++in) {
                          acc_row[in - j0] +=
                              a * cast<float>(B[in * ldb + k]);
                      }
                  }
              }
          }
      }

      // epilogue + store
      for (int im = i0; im < imax; ++im) {
          T* cptr = C + im * ldc + j0;
          float* acc_row = acc_buf[im - i0];

          for (int in = j0; in < jmax; ++in) {
              cptr[in - j0] =
                  epilogue(acc_row[in - j0], im, in);
          }
      }
    }
  }
}