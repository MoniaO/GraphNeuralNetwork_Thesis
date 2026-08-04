#!/usr/bin/env bash
# Wave 3 — zero-vector placebo (same decoder width as compact, features=0).
# 3 seeds; compare to HCR-0 and binary_compact.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="$(date +%Y-%m-%d)"
OUT_DIR="outputs/wave3_hcr_${DAY}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/${DAY}_wave3_HCR_PLACEBO_VISIBLE.log"
CSV="${OUT_DIR}/${DAY}_wave3_HCR_all_runs.csv"

SEEDS=(20260721 20260722 20260723)
VARIANT=all_zero

already_done() {
  local seed="$1"
  python - "$CSV" "$VARIANT" "$seed" <<'PY'
import sys
from pathlib import Path
import pandas as pd
csv, variant, seed = sys.argv[1:4]
p = Path(csv)
if not p.exists():
    raise SystemExit(1)
df = pd.read_csv(p)
m = (
    (df["hcr_variant"].astype(str) == variant)
    & (df["training_seed"].astype(int) == int(seed))
    & df["valid_auprc"].notna()
)
raise SystemExit(0 if m.any() else 1)
PY
}

{
  echo "============================================================"
  echo "WAVE 3 — HCR ZERO-VECTOR PLACEBO"
  echo "Seeds: ${SEEDS[*]}"
  echo "============================================================"
} | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
  if already_done "$seed"; then
    echo "SKIP ${VARIANT} seed=${seed}" | tee -a "$LOG"
    continue
  fi
  echo "===== START ${VARIANT} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python src/train_taskA.py \
    model=TaskA_hgt_hcr \
    hcr=all_zero \
    model.num_layers=1 \
    model.hidden_dim=64 \
    model.hidden_channels=64 \
    model.heads=8 \
    model.dropout=0.2 \
    data.dataset.scenario=clean \
    data.feature_ablation_profile=empirical \
    data.candidate_seed=20260722 \
    training.seed="$seed" \
    training.device=cpu \
    experiment.wave=WAVE3_HCR \
    experiment.intervention=hgt_hcr_all_zero \
    experiment.hcr_variant=all_zero \
    wandb.enabled=true \
    wandb.group=TaskA_WAVE3_HCR \
    wandb.job_type=hcr_placebo \
    'wandb.tags=[TaskA,WAVE3,HCR,PLACEBO,v3,clean,hgt,L1,H8,all_zero]' \
    >>"$LOG" 2>&1
  echo "===== DONE ${VARIANT} seed=${seed} rc=$? $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python scripts/export_wave3_hcr_results.py --day "$DAY" >>"$LOG" 2>&1 || true
done

python scripts/export_wave3_hcr_results.py --day "$DAY" | tee -a "$LOG"
echo "PLACEBO finished $(date)" | tee -a "$LOG"
