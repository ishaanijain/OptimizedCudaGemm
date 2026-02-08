import torch
import cuda_kernels
import time

def benchmark_gemm(fn, M, N, K, warmup=5, iters=20):
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    for _ in range(warmup):
        fn(A, B)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(A, B)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    flops = 2.0 * M * N * K
    gflops = flops / (elapsed_ms * 1e6)
    return elapsed_ms, gflops

def benchmark_transpose(M, N, warmup=5, iters=20):
    x = torch.randn(M, N, device="cuda")
    for _ in range(warmup):
        cuda_kernels.transpose(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        cuda_kernels.transpose(x)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    bw_gb = 2.0 * M * N * 4 / (elapsed_ms * 1e6)  # read + write, float32
    return elapsed_ms, bw_gb

if __name__ == "__main__":
    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]

    kernels = [
        ("naive_gemm", cuda_kernels.naive_gemm),
        ("tiled_gemm", cuda_kernels.tiled_gemm),
        ("reg_blocked", cuda_kernels.register_blocked_gemm),
        ("warp_gemm", cuda_kernels.warp_gemm),
        ("cuBLAS", torch.mm),
    ]

    print("=" * 80)
    print("GEMM Benchmark — GFLOPS (higher is better)")
    print("=" * 80)
    header = f"{'Size':>20s}" + "".join(f"{name:>14s}" for name, _ in kernels)
    print(header)
    print("-" * len(header))

    for M, N, K in sizes:
        row = f"{f'{M}x{N}x{K}':>20s}"
        naive_ms = None
        for name, fn in kernels:
            ms, gflops = benchmark_gemm(fn, M, N, K)
            if name == "naive_gemm":
                naive_ms = ms
            speedup = f" ({ms/naive_ms if naive_ms and naive_ms > 0 else 0:.1f}x)" if naive_ms and name != "naive_gemm" else ""
            row += f"{gflops:>10.1f}{speedup:>4s}"
        print(row)

    print(f"\n{'=' * 60}")
    print("Transpose Benchmark — Effective Bandwidth GB/s")
    print("=" * 60)
    for M, N in [(1024, 1024), (2048, 2048), (4096, 4096)]:
        ms, bw = benchmark_transpose(M, N)
        ms_ref, bw_ref = benchmark_gemm(lambda a, b: a.t().contiguous(), M, N, M)
        print(f"  {M}x{N}: custom={bw:.1f} GB/s ({ms:.3f}ms)  torch.t()={bw_ref:.1f} GB/s ({ms_ref:.3f}ms)")

    print("\nDone.")
