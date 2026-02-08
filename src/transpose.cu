#include <torch/extension.h>
#include "common.h"

// Bank-conflict-free matrix transpose
// Without padding: threads in a warp access same bank on column reads → 32-way conflict
// With +1 padding: stride becomes 33, distributing accesses across all 32 banks → zero conflicts
// Result: ~2-3x speedup over naive transpose

#define TRANS_TILE 32
#define TRANS_PAD 1

__global__ void transpose_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int rows, int cols) {
    // +1 padding eliminates bank conflicts on column access
    __shared__ float tile[TRANS_TILE][TRANS_TILE + TRANS_PAD];

    int xIdx = blockIdx.x * TRANS_TILE + threadIdx.x;
    int yIdx = blockIdx.y * TRANS_TILE + threadIdx.y;

    // Coalesced read from global → shared (row-major)
    if (xIdx < cols && yIdx < rows) {
        tile[threadIdx.y][threadIdx.x] = input[yIdx * cols + xIdx];
    }

    __syncthreads();

    // Transposed indices for output
    xIdx = blockIdx.y * TRANS_TILE + threadIdx.x;
    yIdx = blockIdx.x * TRANS_TILE + threadIdx.y;

    // Coalesced write from shared → global (column read from tile is now bank-conflict-free)
    if (xIdx < rows && yIdx < cols) {
        output[yIdx * rows + xIdx] = tile[threadIdx.x][threadIdx.y];
    }
}

torch::Tensor transpose(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty({cols, rows}, input.options());

    dim3 threads(TRANS_TILE, TRANS_TILE);
    dim3 blocks(CEIL_DIV(cols, TRANS_TILE), CEIL_DIV(rows, TRANS_TILE));

    transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    CUDA_CHECK_KERNEL();

    return output;
}
