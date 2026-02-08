#include <torch/extension.h>
#include "common.h"

// double-buffered, bank-conflict-free GEMM w/ warp shuffles
// +1 pad on smem cols so stride=129 instead of 128 (avoids 32-bank conflicts)

#define WBM 128
#define WBN 128
#define WBK 8
#define WTM 8
#define WTN 8
#define WARP_SIZE 32
#define SMEM_PAD 1

__global__ void warp_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    // ping-pong buffers for overlapping load + compute
    __shared__ float As[2][WBK][WBM + SMEM_PAD];
    __shared__ float Bs[2][WBK][WBN + SMEM_PAD];

    const int numThreads = (WBM / WTM) * (WBN / WTN);
    const int threadRow = threadIdx.x / (WBN / WTN);
    const int threadCol = threadIdx.x % (WBN / WTN);
    const int laneId = threadIdx.x % WARP_SIZE;

    const int blockRowStart = blockIdx.y * WBM;
    const int blockColStart = blockIdx.x * WBN;

    float regC[WTM][WTN] = {0.0f};
    float regA[WTM];
    float regB[WTN];

    int numTiles = CEIL_DIV(K, WBK);
    int curBuf = 0;

    // prefetch first tile
    {
        const int numLoadsA = (WBM * WBK) / numThreads;
        for (int i = 0; i < numLoadsA; ++i) {
            int idx = threadIdx.x + i * numThreads;
            int loadRow = idx / WBK;
            int loadCol = idx % WBK;
            int gRow = blockRowStart + loadRow;
            As[0][loadCol][loadRow] = (gRow < M && loadCol < K) ? A[gRow * K + loadCol] : 0.0f;
        }
        const int numLoadsB = (WBN * WBK) / numThreads;
        for (int i = 0; i < numLoadsB; ++i) {
            int idx = threadIdx.x + i * numThreads;
            int loadRow = idx / WBN;
            int loadCol = idx % WBN;
            int gCol = blockColStart + loadCol;
            Bs[0][loadRow][loadCol] = (loadRow < K && gCol < N) ? B[loadRow * N + gCol] : 0.0f;
        }
    }
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        int nextBuf = 1 - curBuf;

        // prefetch next tile while computing on current
        if (t + 1 < numTiles) {
            int nextK = (t + 1) * WBK;
            const int numLoadsA = (WBM * WBK) / numThreads;
            for (int i = 0; i < numLoadsA; ++i) {
                int idx = threadIdx.x + i * numThreads;
                int loadRow = idx / WBK;
                int loadCol = idx % WBK;
                int gRow = blockRowStart + loadRow;
                int gCol = nextK + loadCol;
                As[nextBuf][loadCol][loadRow] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
            }
            const int numLoadsB = (WBN * WBK) / numThreads;
            for (int i = 0; i < numLoadsB; ++i) {
                int idx = threadIdx.x + i * numThreads;
                int loadRow = idx / WBN;
                int loadCol = idx % WBN;
                int gRow = nextK + loadRow;
                int gCol = blockColStart + loadCol;
                Bs[nextBuf][loadRow][loadCol] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
            }
        }

        #pragma unroll
        for (int k = 0; k < WBK; ++k) {
            #pragma unroll
            for (int tm = 0; tm < WTM; ++tm)
                regA[tm] = As[curBuf][k][threadRow * WTM + tm];

            #pragma unroll
            for (int tn = 0; tn < WTN; ++tn)
                regB[tn] = Bs[curBuf][k][threadCol * WTN + tn];

            // broadcast B within warp via shuffle
            #pragma unroll
            for (int tn = 0; tn < WTN; ++tn)
                regB[tn] = __shfl_sync(0xFFFFFFFF, regB[tn], laneId);

            #pragma unroll
            for (int tm = 0; tm < WTM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < WTN; ++tn)
                    regC[tm][tn] += regA[tm] * regB[tn];
        }

        __syncthreads();
        curBuf = nextBuf;
    }

    for (int tm = 0; tm < WTM; ++tm) {
        for (int tn = 0; tn < WTN; ++tn) {
            int gRow = blockRowStart + threadRow * WTM + tm;
            int gCol = blockColStart + threadCol * WTN + tn;
            if (gRow < M && gCol < N)
                C[gRow * N + gCol] = regC[tm][tn];
        }
    }
}

torch::Tensor warp_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");

    auto C = torch::zeros({M, N}, A.options());

    const int numThreadsPerBlock = (WBM / WTM) * (WBN / WTN);
    dim3 threads(numThreadsPerBlock);
    dim3 blocks(CEIL_DIV(N, WBN), CEIL_DIV(M, WBM));

    warp_gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);
    CUDA_CHECK_KERNEL();

    return C;
}
