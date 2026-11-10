# kernels

FP32 SGEMM in CUDA, written in three steps, from a basic shared-memory tile up to **86.8% of cuBLAS** on an NVIDIA A10.

`C = A * B`, row-major, single precision. Every kernel is checked against cuBLAS for correctness, including sizes that are not multiples of the tile size or of 4.

## Results

NVIDIA A10 (Ampere, SM 8.6, 72 SMs), CUDA 12.8. The baseline is cuBLAS SGEMM in pure FP32 (`CUBLAS_DEFAULT_MATH`, no TF32 tensor cores). Each number is the mean of 10–30 timed launches (CUDA events) after 3 warmup launches. Square N x N matrices.

| N | cuBLAS (GFLOP/s) | `tiled_16x16` | `regtiled_4x4` | `vec_8x8` |
|---|---|---|---|---|
| 1000 | 12,312 | 19.2% | 48.1% | 62.2% |
| 1023 | 12,388 | 19.0% | 48.8% | 50.0% |
| 1024 | 12,720 | 18.9% | 50.0% | 64.0% |
| 2048 | 17,367 | 10.8% | 42.3% | 70.1% |
| 4096 | 12,991 | 13.9% | 54.0% | **86.8%** |

Percentages are throughput relative to cuBLAS. At N = 4096, `vec_8x8` runs in 12.19 ms against 76.12 ms for `tiled_16x16`, which is **6.2x faster**. The maximum relative error against cuBLAS is about 1.6e-6.

## The kernels

Each step increases how much arithmetic is done per value loaded from memory.

### 1. `tiled_16x16`: shared-memory tiling
Each 16x16 thread block loads a 16x16 tile of A and of B into shared memory, so each element read from global memory is reused 16 times. Each thread computes one output, which means **2 shared-memory loads per FMA**. That makes shared-memory bandwidth the limit.

### 2. `regtiled_4x4`: register tiling
Each 256-thread block computes a 64x64 tile of C, and each thread keeps a **4x4 block of outputs in registers**. Per k-step, a thread loads 4 values of A and 4 of B and performs 16 FMAs. That's 8 loads for 16 FMAs, 4x the arithmetic per load of step 1. The A tile is stored transposed in shared memory so that each thread reads a contiguous column.

### 3. `vec_8x8`: larger register tiles and vectorized memory access
Each block computes a 128x128 tile and each thread an **8x8 output block**: 16 loads feed 64 FMAs. Global and shared memory accesses use `float4` (16 bytes per instruction), and the larger block tile reuses more data from global memory. There's a scalar fallback when K or N is not a multiple of 4, which is why N = 1023 is slower. It is built without `--maxrregcount`, so the 64 accumulators stay in registers with no spills.

### What cuBLAS does that these kernels don't (yet)
- Double buffering, so the next tile's load overlaps with the current tile's compute.
- Warp-level tiling, which reduces shared-memory bank conflicts.
- Tuning for each architecture and problem size.

## Build and run

You need the CUDA toolkit (nvcc) and cuBLAS.

```bash
make bench
```

`bench/bench_gemm.cu` times each kernel against cuBLAS for N = 1000, 1023, 1024, 2048 and 4096, and exits non-zero if any kernel's output differs from cuBLAS.

## Layout

```
src/gemm.cuh         launcher declarations
src/gemm.cu          the three kernels and their launchers
bench/bench_gemm.cu  benchmark + correctness check against cuBLAS
```
