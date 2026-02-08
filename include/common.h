#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel launch error checking
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "Kernel launch error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Ceiling division
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// GPU timer utility
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic() { cudaEventRecord(start, 0); }

    float toc() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed; // milliseconds
    }
};
