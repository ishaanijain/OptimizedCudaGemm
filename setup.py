from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")

setup(
    name="cuda_kernels",
    version="1.0.0",
    description="High-Performance CUDA Kernels for Matrix Operations",
    ext_modules=[
        CUDAExtension(
            name="cuda_kernels",
            sources=[
                "src/bindings.cpp",
                "src/naive_gemm.cu",
                "src/tiled_gemm.cu",
                "src/register_blocked_gemm.cu",
                "src/warp_gemm.cu",
                "src/transpose.cu",
            ],
            include_dirs=[include_dir],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_70,code=sm_70",  # Volta
                    "-gencode=arch=compute_75,code=sm_75",  # Turing
                    "-gencode=arch=compute_80,code=sm_80",  # Ampere
                    "-gencode=arch=compute_86,code=sm_86",  # Ampere consumer
                    "-gencode=arch=compute_89,code=sm_89",  # Ada Lovelace
                    "-gencode=arch=compute_90,code=sm_90",  # Hopper
                    "--ptxas-options=-v",  # Show register/shared mem usage
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
