#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// FP32 row-major GEMM launchers: C (M x N) = A (M x K) * B (K x N). All pointers are device memory.

// Baseline: 16x16 shared-memory tiles, one output element per thread.
void launch_gemm(float* C, const float* A, const float* B, size_t M, size_t N, size_t K, cudaStream_t stream = 0);

// Register-tiled: 64x64 block tile, 4x4 outputs per thread.
void launch_gemm_regtiled(float* C, const float* A, const float* B, size_t M, size_t N, size_t K, cudaStream_t stream = 0);

// Vectorized register-tiled: 128x128 block tile, 8x8 outputs per thread, float4 loads/stores.
void launch_gemm_vec(float* C, const float* A, const float* B, size_t M, size_t N, size_t K, cudaStream_t stream = 0);
