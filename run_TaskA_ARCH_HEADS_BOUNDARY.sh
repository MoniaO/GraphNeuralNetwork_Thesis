#!/usr/bin/env bash
# Wave 2 / HEADS BOUNDARY — HGT L1 heads=16 vs locked H8 (FINAL).
# Choose H16 only if mean paired Δ valid AUPRC (H16 − H8) > 0; else keep H8.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="$(date +%Y-%m-%d)"
OUT_DIR="outputs/wave2_architecture_${DAY}_HEADS_BOUNDARY"
mkdir -p "$OUT_DIR" "$OUT_DIR/by_model"
LOG="${OUT_DIR}/${DAY}_wave2_ARCH_HEADS_BOUNDARY_VISIBLE.log"
CSV="${OUT_DIR}/${DAY}_wave2_ARCH_HEADS_BOUNDARY_all_runs.csv"

SEEDS=(20260721 20260722 20260723 20260724 20260725)

already_done() {
  local seed="$1"
  python - "$CSV" "$seed" <<'PY'
import sys
from pathlib import Path
import pandas as pd
csv, seed = sys.argv[1:3]
p = Path(csv)
if not p.exists():
    raise SystemExit(1)
df = pd.read_csv(p)
m = (
    (df["model"].astype(str) == "hgt")
    & (df["num_layers"].astype(int) == 1)
    & (df["heads"].astype(int) == 16)
    & (df["training_seed"].astype(int) == int(seed))
    & df["valid_auprc"].notna()
)
raise SystemExit(0 if m.any() else 1)
PY
}

{
  echo "============================================================"
  echo "WAVE 2 — ARCH HEADS BOUNDARY (H16)"
  echo "Date: ${DAY}"
  echo "HGT L1 | hidden=64 | heads=16 | clean | empirical | cpu"
  echo "Seeds: ${SEEDS[*]}"
  echo "Rule: choose H16 iff mean(Δ valid AUPRC H16−H8) > 0"
  echo "============================================================"
} | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
  if already_done "$seed"; then
    echo "SKIP hgt_L1_H16 seed=${seed}" | tee -a "$LOG"
    continue
  fi
  echo "===== START hgt_L1_H16 seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python src/train_taskA.py \
    model=TaskA_hgt \
    model.num_layers=1 \
    model.hidden_dim=64 \
    model.hidden_channels=64 \
    model.heads=16 \
    model.dropout=0.2 \
    data.dataset.scenario=clean \
    data.feature_ablation_profile=empirical \
    data.feature_ablation_seed=20260722 \
    data.candidate_seed=20260722 \
    training.seed="$seed" \
    training.device=cpu \
    experiment.wave=ARCH_HEADS_BOUNDARY \
    experiment.intervention=hgt_L1_H16 \
    experiment.hcr_variant=none \
    wandb.enabled=true \
    wandb.group=TaskA_ARCH_HEADS_BOUNDARY \
    wandb.job_type=architecture_heads_boundary \
    'wandb.tags=[TaskA,WAVE2,ARCH_HEADS_BOUNDARY,v3,clean,no_hcr,hgt,L1,H16]' \
    >>"$LOG" 2>&1
  rc=$?
  echo "===== DONE hgt_L1_H16 seed=${seed} rc=${rc} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_HEADS_BOUNDARY >>"$LOG" 2>&1 || true
done

python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_HEADS_BOUNDARY >>"$LOG" 2>&1 || true
python scripts/decide_wave2_h16_boundary.py --day "$DAY" >>"$LOG" 2>&1 || true

echo "ARCH_HEADS_BOUNDARY finished $(date) → ${OUT_DIR}" | tee -a "$LOG"
