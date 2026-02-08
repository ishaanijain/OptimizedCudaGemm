#!/bin/bash
# nsight-compute profiling for CUDA kernel analysis
# Usage: bash ncu_scripts/profile.sh
# Requires: nsight-compute (ncu) installed and in PATH

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_DIR/ncu_reports"
mkdir -p "$REPORT_DIR"

PROFILE_SCRIPT=$(cat <<'PYEOF'
import torch, cuda_kernels
M, N, K = 2048, 2048, 2048
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")
cuda_kernels.naive_gemm(A, B)
cuda_kernels.tiled_gemm(A, B)
cuda_kernels.register_blocked_gemm(A, B)
cuda_kernels.warp_gemm(A, B)
cuda_kernels.transpose(A)
PYEOF
)

echo "=== Profiling all kernels ==="

# Full metrics collection
ncu --set full \
    --target-processes all \
    --export "$REPORT_DIR/full_report" \
    python -c "$PROFILE_SCRIPT"

echo ""
echo "=== Key metrics summary ==="

# Lightweight summary: occupancy, memory throughput, compute throughput
ncu --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
    python -c "$PROFILE_SCRIPT"

echo ""
echo "Reports saved to: $REPORT_DIR/"
echo "Open with: ncu-ui $REPORT_DIR/full_report.ncu-rep"
