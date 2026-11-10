NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++20 -arch=native

BENCH = bench/bench_gemm

.PHONY: all bench clean

all: $(BENCH)

$(BENCH): src/gemm.cu src/gemm.cuh bench/bench_gemm.cu
	$(NVCC) $(NVCCFLAGS) src/gemm.cu bench/bench_gemm.cu -lcublas -o $@

bench: $(BENCH)
	./$(BENCH)

clean:
	rm -f $(BENCH)
