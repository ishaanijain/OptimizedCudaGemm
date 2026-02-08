#include <torch/extension.h>

// Forward declarations
torch::Tensor naive_gemm(torch::Tensor A, torch::Tensor B);
torch::Tensor tiled_gemm(torch::Tensor A, torch::Tensor B);
torch::Tensor register_blocked_gemm(torch::Tensor A, torch::Tensor B);
torch::Tensor warp_gemm(torch::Tensor A, torch::Tensor B);
torch::Tensor transpose(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_gemm", &naive_gemm, "Naive GEMM (baseline)");
    m.def("tiled_gemm", &tiled_gemm, "Shared-memory tiled GEMM");
    m.def("register_blocked_gemm", &register_blocked_gemm, "Register-blocked GEMM with coalescing");
    m.def("warp_gemm", &warp_gemm, "Warp-optimized GEMM (flagship)");
    m.def("transpose", &transpose, "Bank-conflict-free transpose");
}
