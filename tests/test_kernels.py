import torch
import cuda_kernels

def test_correctness(fn, name, sizes=None):
    if sizes is None:
        sizes = [(128, 128, 128), (256, 512, 256), (1024, 1024, 1024),
                 (100, 200, 300), (513, 257, 129)]  # Non-aligned sizes
    passed = 0
    for M, N, K in sizes:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        ref = torch.mm(A, B)
        out = fn(A, B)
        if torch.allclose(out, ref, atol=1e-3, rtol=1e-3):
            passed += 1
        else:
            maxdiff = (out - ref).abs().max().item()
            print(f"  FAIL {name} ({M}x{N}x{K}): max diff = {maxdiff:.6f}")
    print(f"  {name}: {passed}/{len(sizes)} passed")
    return passed == len(sizes)

def test_transpose():
    sizes = [(128, 128), (256, 512), (1024, 1024), (100, 200), (513, 257)]
    passed = 0
    for R, C in sizes:
        x = torch.randn(R, C, device="cuda")
        ref = x.t().contiguous()
        out = cuda_kernels.transpose(x)
        if torch.allclose(out, ref, atol=1e-5):
            passed += 1
        else:
            maxdiff = (out - ref).abs().max().item()
            print(f"  FAIL transpose ({R}x{C}): max diff = {maxdiff:.6f}")
    print(f"  transpose: {passed}/{len(sizes)} passed")
    return passed == len(sizes)

if __name__ == "__main__":
    print("=== CUDA Kernels Correctness Tests ===\n")
    all_pass = True
    for fn, name in [
        (cuda_kernels.naive_gemm, "naive_gemm"),
        (cuda_kernels.tiled_gemm, "tiled_gemm"),
        (cuda_kernels.register_blocked_gemm, "register_blocked_gemm"),
        (cuda_kernels.warp_gemm, "warp_gemm"),
    ]:
        all_pass &= test_correctness(fn, name)
    all_pass &= test_transpose()
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
