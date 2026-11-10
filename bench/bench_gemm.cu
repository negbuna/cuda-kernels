#include "../src/gemm.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

// Benchmarks the tiled and register-tiled GEMM kernels against cuBLAS SGEMM (FP32, no TF32).
// All matrices are row-major: C (M x N) = A (M x K) * B (K x N).

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while (0)
#define CUBLAS_CHECK(call) do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(s), __FILE__, __LINE__); std::exit(1); } } while (0)

static float time_ms(const std::function<void()>& launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / iters;
}

// Max element-wise error relative to the largest reference magnitude
static double rel_error(const std::vector<float>& out, const std::vector<float>& ref) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(out[i] - ref[i])));
        max_ref = std::max(max_ref, static_cast<double>(std::fabs(ref[i])));
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)); // pure FP32, no TF32 tensor cores

    const std::vector<int> sizes = {1000, 1023, 1024, 2048, 4096};
    std::printf("%-6s %-14s %10s %10s %10s %12s\n", "N", "kernel", "ms", "GFLOP/s", "% cuBLAS", "rel_err");

    bool all_correct = true;
    for (int n : sizes) {
        const size_t M = n, N = n, K = n;
        std::vector<float> h_A(M * K), h_B(K * N), h_ref(M * N), h_out(M * N);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : h_A) v = dist(rng);
        for (auto& v : h_B) v = dist(rng);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        const float alpha = 1.0f, beta = 0.0f;
        // Row-major C = A*B is column-major C^T = B^T * A^T
        auto run_cublas = [&] {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        };
        auto run_tiled = [&] { launch_gemm(d_C, d_A, d_B, M, N, K); };
        auto run_regtiled = [&] { launch_gemm_regtiled(d_C, d_A, d_B, M, N, K); };
        auto run_vec = [&] { launch_gemm_vec(d_C, d_A, d_B, M, N, K); };

        const int warmup = 3;
        const int iters = n >= 4096 ? 10 : 30;
        const double flops = 2.0 * M * N * K;

        run_cublas();
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float cublas_ms = time_ms(run_cublas, warmup, iters);
        std::printf("%-6d %-14s %10.3f %10.1f %10.1f %12s\n", n, "cublas_sgemm", cublas_ms, flops / cublas_ms / 1e6, 100.0, "ref");

        struct Candidate { const char* name; std::function<void()> launch; };
        for (const Candidate& c : {Candidate{"tiled_16x16", run_tiled}, Candidate{"regtiled_4x4", run_regtiled}, Candidate{"vec_8x8", run_vec}}) {
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
            c.launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
            double err = rel_error(h_out, h_ref);
            bool ok = err < 1e-4;
            all_correct = all_correct && ok;
            float ms = time_ms(c.launch, warmup, iters);
            std::printf("%-6d %-14s %10.3f %10.1f %10.1f %12.2e%s\n", n, c.name, ms, flops / ms / 1e6,
                        100.0 * cublas_ms / ms, err, ok ? "" : "  MISMATCH");
        }
        std::printf("\n");

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    std::printf(all_correct ? "All kernels match cuBLAS.\n" : "ERROR: at least one kernel does not match cuBLAS.\n");
    return all_correct ? 0 : 1;
}
