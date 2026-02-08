#include <torch/extension.h>
#include "common.h"

// smem transpose w/ +1 padding to avoid bank conflicts

#define TRANS_TILE 32
#define TRANS_PAD 1

__global__ void transpose_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int rows, int cols) {
    __shared__ float tile[TRANS_TILE][TRANS_TILE + TRANS_PAD];

    int xIdx = blockIdx.x * TRANS_TILE + threadIdx.x;
    int yIdx = blockIdx.y * TRANS_TILE + threadIdx.y;

    if (xIdx < cols && yIdx < rows) {
        tile[threadIdx.y][threadIdx.x] = input[yIdx * cols + xIdx];
    }

    __syncthreads();

    xIdx = blockIdx.y * TRANS_TILE + threadIdx.x;
    yIdx = blockIdx.x * TRANS_TILE + threadIdx.y;

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
