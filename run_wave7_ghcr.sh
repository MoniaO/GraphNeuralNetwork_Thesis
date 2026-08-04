#!/usr/bin/env bash
# Wave 7 GHCR — frozen L3_H8_W32_leaky + V0/V1/V2
# Usage:
#   ./run_wave7_ghcr.sh pilots      # 3 clean variants
#   ./run_wave7_ghcr.sh scenarios  # all scenario × variant
#   ./run_wave7_ghcr.sh all
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1

STAGE="${1:-pilots}"
mkdir -p outputs/wave7
LOG="outputs/wave7/wave7_${STAGE}.log"
echo "Wave7 stage=${STAGE} → ${LOG}"
python scripts/run_wave7_ghcr.py --stage "${STAGE}" --epochs "${EPOCHS:-300}" 2>&1 | tee -a "${LOG}"
