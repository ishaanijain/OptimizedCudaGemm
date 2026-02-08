# cuda-kernels

Optimized CUDA GEMM kernels, built as a PyTorch C++ extension.

Started with a naive matmul and worked my way up through shared memory tiling, register blocking, and warp-level optimizations. Each kernel builds on the last one's bottlenecks — profiled everything with nsight-compute to figure out what was actually slow.

## what's here

- **naive** — one thread per output element, pure global memory. baseline.
- **tiled** — 32x32 shared memory tiles. huge win from data reuse (~3x over naive)
- **register-blocked** — 8x8 micro-tiles per thread, outer product in registers. cuts bandwidth ~40% since you're not hammering shared mem as hard
- **warp-optimized** — the good stuff: double buffering to hide latency, +1 padding to kill bank conflicts, warp shuffles. this is the one that actually gets close to cuBLAS
- **transpose** — mostly wrote this to isolate the bank conflict thing. +1 padding trick makes a huge difference

## build

needs CUDA toolkit + PyTorch with CUDA support

```
pip install -e .
```

## run

```
python tests/test_kernels.py        # correctness check against torch.mm
python benchmarks/benchmark.py      # gflops comparison across all kernels
```

## profiling

```
bash ncu_scripts/profile.sh
```

generates nsight-compute reports — look at `sm__throughput`, `dram__throughput`, and `l1tex__data_bank_conflicts` to see the progression across kernel versions.

## usage in python

```python
import cuda_kernels

C = cuda_kernels.warp_gemm(A, B)       # fastest
C = cuda_kernels.tiled_gemm(A, B)      # simpler, still fast
T = cuda_kernels.transpose(X)          # bank-conflict-free
```

all kernels handle non-power-of-2 sizes.
