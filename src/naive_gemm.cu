#include <torch/extension.h>
#include "common.h"

// Naive GEMM: one thread computes one output element
// No data reuse — each thread loads full row of A and column of B from global memory
// Arithmetic intensity: O(1) — memory bound baseline
__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor naive_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A columns != B rows");

    auto C = torch::zeros({M, N}, A.options());

    const int BLOCK = 16;
    dim3 threads(BLOCK, BLOCK);
    dim3 blocks(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK));

    naive_gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);
    CUDA_CHECK_KERNEL();

    return C;
}
