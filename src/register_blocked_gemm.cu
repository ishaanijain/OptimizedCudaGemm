#include <torch/extension.h>
#include "common.h"

// each thread computes a TM x TN micro-tile in registers
// A is stored transposed in smem to avoid bank conflicts on col access

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

__global__ void register_blocked_gemm_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K) {
    __shared__ float As[BK][BM]; // transposed
    __shared__ float Bs[BK][BN];

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    const int blockRowStart = blockIdx.y * BM;
    const int blockColStart = blockIdx.x * BN;

    A += blockRowStart * K;
    B += blockColStart;
    C += blockRowStart * N + blockColStart;

    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    const int numThreads = (BM / TM) * (BN / TN);
    const int numLoadsA = (BM * BK) / numThreads;
    const int numLoadsB = (BN * BK) / numThreads;

    for (int bk = 0; bk < K; bk += BK) {
        // load A tile (transposed into smem)
        for (int i = 0; i < numLoadsA; ++i) {
            int linearIdx = threadIdx.x + i * numThreads;
            int loadRow = linearIdx / BK;
            int loadCol = linearIdx % BK;
            int globalRow = blockRowStart + loadRow;
            int globalCol = bk + loadCol;
            As[loadCol][loadRow] = (globalRow < M && globalCol < K)
                                       ? A[loadRow * K + bk + loadCol]
                                       : 0.0f;
        }

        // load B tile
        for (int i = 0; i < numLoadsB; ++i) {
            int linearIdx = threadIdx.x + i * numThreads;
            int loadRow = linearIdx / BN;
            int loadCol = linearIdx % BN;
            int globalRow = bk + loadRow;
            int globalCol = blockColStart + loadCol;
            Bs[loadRow][loadCol] = (globalRow < K && globalCol < N)
                                       ? B[(bk + loadRow) * N + blockColStart + loadCol]
                                       : 0.0f;
        }

        __syncthreads();

        // outer product in registers
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                regA[tm] = As[k][threadRow * TM + tm];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn)
                regB[tn] = Bs[k][threadCol * TN + tn];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn)
                    regC[tm][tn] += regA[tm] * regB[tn];
        }

        __syncthreads();
    }

    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            int globalRow = threadRow * TM + tm;
            int globalCol = threadCol * TN + tn;
            if ((blockRowStart + globalRow) < M && (blockColStart + globalCol) < N) {
                C[globalRow * N + globalCol] = regC[tm][tn];
            }
        }
    }
}

torch::Tensor register_blocked_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");

    auto C = torch::zeros({M, N}, A.options());

    const int numThreadsPerBlock = (BM / TM) * (BN / TN);
    dim3 threads(numThreadsPerBlock);
    dim3 blocks(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    register_blocked_gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);
    CUDA_CHECK_KERNEL();

    return C;
}
