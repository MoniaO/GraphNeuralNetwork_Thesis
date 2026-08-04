#!/usr/bin/env bash
# Wave 5D — L2_structural_hcr architecture audit (NOT Wave2 ARCH_DEPTH).
# Usage:
#   ./run_TaskA_L2_ARCH_AUDIT.sh            # pipeline check / print Stage2 grid
#   ./run_TaskA_L2_ARCH_AUDIT.sh 0          # Stage 0 reproduce (5 seeds)
#   ./run_TaskA_L2_ARCH_AUDIT.sh 1          # Stage 1 screening (63 runs)
#   ./run_TaskA_L2_ARCH_AUDIT.sh 2          # Stage 2 joint (16×5=80) → CSV + W&B
#   ./run_TaskA_L2_ARCH_AUDIT.sh 3          # Stage 3 top-3 final (3×5=15)
#   ./run_TaskA_L2_ARCH_AUDIT.sh 23         # Stage 2 then Stage 3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"
export WANDB_MODE="${WANDB_MODE:-online}"

STAGE="${1:-check}"
mkdir -p outputs/taskA_arch_audit
LOG="outputs/taskA_arch_audit/stage${STAGE}_console.log"

echo "============================================================"
echo "WAVE 5D L2 ARCH AUDIT — stage=${STAGE}"
echo "Do NOT use ./run_TaskA_ARCH_DEPTH.sh for L2 (that is Wave2/no_hcr)."
echo "Log: ${LOG}"
echo "============================================================"

python scripts/run_taskA_arch_audit.py --stage "$STAGE" 2>&1 | tee "$LOG"
