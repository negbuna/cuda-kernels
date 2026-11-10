#include "gemm.cuh"
#include <stdexcept>
#include <string>

#define TILE_SIZE 16

// CUDA Kernel for Tiled GEMM
// blockDim.x = 16, blockDim.y = 16. Total threads per block = 256.
// __launch_bounds__(256) caps register usage and guarantees occupancy.
__global__ void __launch_bounds__(256) tiled_gemm_kernel(
    float* C, const float* A, const float* B,
    int M, int N, int K) 
{
    // Shared memory for tiles of A and B
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float c_val = 0.0f;

    // Loop over all the tiles of A and B required to compute the block
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            s_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        if (col < N && (t * TILE_SIZE + ty) < K) {
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their elements
        __syncthreads();

        // Multiply the two matrices' tiles together
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            c_val += s_A[ty][k] * s_B[k][tx];
        }

        // Synchronize to ensure all threads are done computing before loading next tile
        __syncthreads();
    }

    // Write computed element to global memory
    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}


// Host wrapper to launch the tiled GEMM CUDA kernel
void launch_gemm(float* C, const float* A, const float* B, size_t M, size_t N, size_t K, cudaStream_t stream) {
    if (M == 0 || N == 0 || K == 0) return;

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(C, A, B, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to launch tiled_gemm_kernel: ") + cudaGetErrorString(err));
    }
}

// Register-tiled GEMM: each 256-thread block computes a 64x64 tile of C, each thread a 4x4 sub-tile.
// A tile is stored transposed in shared memory so each thread reads a contiguous column of A per k-step.
#define RT_BM 64
#define RT_BN 64
#define RT_BK 8
#define RT_TM 4
#define RT_TN 4
__global__ void __launch_bounds__(256) regtiled_gemm_kernel(
    float* C, const float* A, const float* B,
    int M, int N, int K)
{
    __shared__ float s_A[RT_BK][RT_BM];
    __shared__ float s_B[RT_BK][RT_BN];

    int tid = threadIdx.x;
    int t_row = tid / (RT_BN / RT_TN);
    int t_col = tid % (RT_BN / RT_TN);
    int block_row = blockIdx.y * RT_BM;
    int block_col = blockIdx.x * RT_BN;

    float acc[RT_TM][RT_TN] = {};
    float a_reg[RT_TM];
    float b_reg[RT_TN];

    for (int k0 = 0; k0 < K; k0 += RT_BK) {
        // Load A tile (BM x BK) transposed into s_A
        for (int i = tid; i < RT_BM * RT_BK; i += 256) {
            int r = i / RT_BK, c = i % RT_BK;
            int g_r = block_row + r, g_c = k0 + c;
            s_A[c][r] = (g_r < M && g_c < K) ? A[g_r * K + g_c] : 0.0f;
        }
        // Load B tile (BK x BN) into s_B
        for (int i = tid; i < RT_BK * RT_BN; i += 256) {
            int r = i / RT_BN, c = i % RT_BN;
            int g_r = k0 + r, g_c = block_col + c;
            s_B[r][c] = (g_r < K && g_c < N) ? B[g_r * N + g_c] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < RT_BK; ++k) {
            #pragma unroll
            for (int i = 0; i < RT_TM; ++i) a_reg[i] = s_A[k][t_row * RT_TM + i];
            #pragma unroll
            for (int j = 0; j < RT_TN; ++j) b_reg[j] = s_B[k][t_col * RT_TN + j];
            #pragma unroll
            for (int i = 0; i < RT_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < RT_TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RT_TM; ++i) {
        #pragma unroll
        for (int j = 0; j < RT_TN; ++j) {
            int r = block_row + t_row * RT_TM + i;
            int c = block_col + t_col * RT_TN + j;
            if (r < M && c < N) {
                C[r * N + c] = acc[i][j];
            }
        }
    }
}

void launch_gemm_regtiled(float* C, const float* A, const float* B, size_t M, size_t N, size_t K, cudaStream_t stream) {
    if (M == 0 || N == 0 || K == 0) return;

    dim3 blockDim(256);
    dim3 gridDim((N + RT_BN - 1) / RT_BN, (M + RT_BM - 1) / RT_BM);

    regtiled_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(C, A, B, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to launch regtiled_gemm_kernel: ") + cudaGetErrorString(err));
    }
}

// Vectorized register-tiled GEMM: each 256-thread block computes a 128x128 tile of C, each thread an 8x8 sub-tile.
// Global loads/stores use float4 when rows are 16-byte aligned (K and N multiples of 4), with scalar fallback at edges.
// Uses a large register budget per thread: build without --maxrregcount so the accumulators do not spill.
#define VT_BM 128
#define VT_BN 128
#define VT_BK 8
#define VT_TM 8
#define VT_TN 8
__global__ void __launch_bounds__(256) vec_gemm_kernel(
    float* C, const float* A, const float* B,
    int M, int N, int K)
{
    __shared__ __align__(16) float s_A[VT_BK][VT_BM];
    __shared__ __align__(16) float s_B[VT_BK][VT_BN];

    int tid = threadIdx.x;
    int t_row = tid / (VT_BN / VT_TN);
    int t_col = tid % (VT_BN / VT_TN);
    int block_row = blockIdx.y * VT_BM;
    int block_col = blockIdx.x * VT_BN;
    bool a_vec = (K % 4) == 0;
    bool b_vec = (N % 4) == 0;

    // Each thread loads 4 consecutive A elements (one row, 4 k's) and 4 consecutive B elements (one k, 4 cols)
    int a_r = tid / 2, a_c = (tid % 2) * 4;
    int b_r = tid / 32, b_c = (tid % 32) * 4;

    float acc[VT_TM][VT_TN] = {};
    float a_reg[VT_TM];
    float b_reg[VT_TN];

    for (int k0 = 0; k0 < K; k0 += VT_BK) {
        int g_r = block_row + a_r, g_c = k0 + a_c;
        if (a_vec && g_r < M && g_c + 3 < K) {
            float4 v = *reinterpret_cast<const float4*>(&A[g_r * K + g_c]);
            s_A[a_c + 0][a_r] = v.x; s_A[a_c + 1][a_r] = v.y; s_A[a_c + 2][a_r] = v.z; s_A[a_c + 3][a_r] = v.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s_A[a_c + i][a_r] = (g_r < M && g_c + i < K) ? A[g_r * K + g_c + i] : 0.0f;
            }
        }

        g_r = k0 + b_r; g_c = block_col + b_c;
        if (b_vec && g_r < K && g_c + 3 < N) {
            *reinterpret_cast<float4*>(&s_B[b_r][b_c]) = *reinterpret_cast<const float4*>(&B[g_r * N + g_c]);
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s_B[b_r][b_c + i] = (g_r < K && g_c + i < N) ? B[g_r * N + g_c + i] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < VT_BK; ++k) {
            *reinterpret_cast<float4*>(&a_reg[0]) = *reinterpret_cast<const float4*>(&s_A[k][t_row * VT_TM]);
            *reinterpret_cast<float4*>(&a_reg[4]) = *reinterpret_cast<const float4*>(&s_A[k][t_row * VT_TM + 4]);
            *reinterpret_cast<float4*>(&b_reg[0]) = *reinterpret_cast<const float4*>(&s_B[k][t_col * VT_TN]);
            *reinterpret_cast<float4*>(&b_reg[4]) = *reinterpret_cast<const float4*>(&s_B[k][t_col * VT_TN + 4]);
            #pragma unroll
            for (int i = 0; i < VT_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < VT_TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < VT_TM; ++i) {
        int r = block_row + t_row * VT_TM + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < VT_TN; j += 4) {
            int c = block_col + t_col * VT_TN + j;
            if (b_vec && c + 3 < N) {
                *reinterpret_cast<float4*>(&C[r * N + c]) = make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
            } else {
                for (int jj = 0; jj < 4; ++jj) {
                    if (c + jj < N) C[r * N + c + jj] = acc[i][j + jj];
                }
            }
        }
    }
}

void launch_gemm_vec(float* C, const float* A, const float* B, size_t M, size_t N, size_t K, cudaStream_t stream) {
    if (M == 0 || N == 0 || K == 0) return;

    dim3 blockDim(256);
    dim3 gridDim((N + VT_BN - 1) / VT_BN, (M + VT_BM - 1) / VT_BM);

    vec_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(C, A, B, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to launch vec_gemm_kernel: ") + cudaGetErrorString(err));
    }
}
