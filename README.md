# High-Performance CUDA Kernels for Matrix Operations

C++ / CUDA / Python / PyTorch

## Overview

Progressive optimization of GEMM (General Matrix Multiply) kernels demonstrating key GPU programming techniques. Integrated as a PyTorch C++ extension and validated with NVIDIA Nsight Compute.

## Key Results

| Optimization | Technique | Improvement |
|---|---|---|
| Shared memory tiling | Tile-based data reuse, coalesced loads | **3x+ throughput** vs naive |
| Register blocking | TM×TN micro-tiles, outer product in registers | **40%+ bandwidth reduction** |
| Warp primitives | `__shfl_sync`, bank-conflict-free padding, double buffering | Highest throughput |

## Kernels

### 1. Naive GEMM (`naive_gemm.cu`)
Baseline — one thread per output element, global memory only. Memory-bound with O(1) arithmetic intensity.

### 2. Tiled GEMM (`tiled_gemm.cu`)
32×32 shared memory tiles. Each element reused 32× within a tile, reducing global memory traffic by ~32×. Uses `__syncthreads()` barriers with boundary checks for non-aligned matrices.

### 3. Register-Blocked GEMM (`register_blocked_gemm.cu`)
Each thread computes an 8×8 micro-tile entirely in registers. Coalesced global→shared loads, shared→register staging. Transposed A-tile layout in shared memory avoids bank conflicts on column access.

### 4. Warp-Optimized GEMM (`warp_gemm.cu`) — Flagship
- **Bank-conflict elimination**: +1 column padding (stride 129 vs 128)
- **Double buffering**: Overlaps next tile prefetch with current tile compute
- **Warp shuffle**: `__shfl_sync` for cross-lane value broadcast
- **128×128 block tiles** with 8×8 per-thread micro-tiles (256 threads/block)

### 5. Bank-Conflict-Free Transpose (`transpose.cu`)
Shared memory transpose with +1 padding. Without padding: 32-way bank conflicts on column reads. With padding: zero conflicts, ~2-3× speedup.

## Build & Run

```bash
# Install as PyTorch extension
pip install -e .

# Run correctness tests
python tests/test_kernels.py

# Run benchmarks
python benchmarks/benchmark.py

# Profile with nsight-compute
bash ncu_scripts/profile.sh
```

## Architecture

```
A (M×K) × B (K×N) → C (M×N)

Global Memory ──→ Shared Memory Tile ──→ Registers ──→ Output
  (coalesced)      (bank-conflict-free)    (TM×TN)     (coalesced)
                   ↕ double buffered
```

## Requirements

- CUDA Toolkit 11.0+
- PyTorch 1.9+ with CUDA support
- Python 3.8+
- GPU: Volta (SM 70) or newer
