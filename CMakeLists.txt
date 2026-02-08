cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

find_package(CUDAToolkit REQUIRED)

# Standalone test binary (independent of PyTorch)
add_executable(test_standalone
    src/naive_gemm.cu
    src/tiled_gemm.cu
    src/register_blocked_gemm.cu
    src/warp_gemm.cu
    src/transpose.cu
)

target_include_directories(test_standalone PRIVATE include)
target_compile_options(test_standalone PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --ptxas-options=-v>)
target_link_libraries(test_standalone CUDA::cudart)
