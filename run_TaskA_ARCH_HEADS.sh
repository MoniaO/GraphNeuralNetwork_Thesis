#!/usr/bin/env bash
# Wave 2 / HEADS — HGT attention-head sweep (after DEPTH).
#   HGT_LAYERS=1 ./run_TaskA_ARCH_HEADS.sh
# Matrix: heads ∈ {2,4,8} × 3 seeds = 9 runs
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="$(date +%Y-%m-%d)"
OUT_DIR="outputs/wave2_architecture_${DAY}_HEADS"
mkdir -p "$OUT_DIR" "$OUT_DIR/by_model"
LOG="${OUT_DIR}/${DAY}_wave2_ARCH_HEADS_VISIBLE.log"
CSV="${OUT_DIR}/${DAY}_wave2_ARCH_HEADS_all_runs.csv"

HGT_LAYERS="${HGT_LAYERS:-1}"
SEEDS=(20260721 20260722 20260723)
HEADS=(2 4 8)

already_done() {
  local layers="$1" heads="$2" seed="$3"
  python - "$CSV" "$layers" "$heads" "$seed" <<'PY'
import sys
from pathlib import Path
import pandas as pd
csv, layers, heads, seed = sys.argv[1:5]
p = Path(csv)
if not p.exists():
    raise SystemExit(1)
df = pd.read_csv(p)
m = (
    (df["model"].astype(str) == "hgt")
    & (df["num_layers"].astype(int) == int(layers))
    & (df["heads"].astype(int) == int(heads))
    & (df["training_seed"].astype(int) == int(seed))
    & df["valid_auprc"].notna()
)
raise SystemExit(0 if m.any() else 1)
PY
}

{
  echo "============================================================"
  echo "WAVE 2 — ARCH HEADS (robust)"
  echo "Date: ${DAY}"
  echo "HGT layers: ${HGT_LAYERS}"
  echo "Heads: ${HEADS[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "device=cpu | skip finished"
  echo "============================================================"
} | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
  for heads in "${HEADS[@]}"; do
    label="hgt_L${HGT_LAYERS}_H${heads}"
    if already_done "$HGT_LAYERS" "$heads" "$seed"; then
      echo "SKIP ${label} seed=${seed}" | tee -a "$LOG"
      continue
    fi
    echo "===== START ${label} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    python src/train_taskA.py \
      model=TaskA_hgt \
      model.num_layers="$HGT_LAYERS" \
      model.heads="$heads" \
      data.dataset.scenario=clean \
      data.feature_ablation_profile=empirical \
      data.candidate_seed=20260722 \
      training.seed="$seed" \
      training.device=cpu \
      experiment.wave=ARCH_HEADS \
      experiment.intervention="heads_${label}" \
      experiment.hcr_variant=none \
      wandb.enabled=true \
      wandb.group=TaskA_ARCH_HEADS \
      wandb.job_type=architecture_heads \
      "wandb.tags=[TaskA,WAVE2,ARCH_HEADS,v3,clean,no_hcr,hgt,L${HGT_LAYERS},H${heads}]" \
      >>"$LOG" 2>&1
    rc=$?
    echo "===== DONE ${label} seed=${seed} rc=${rc} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_HEADS >>"$LOG" 2>&1 || true
  done
done

python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_HEADS >>"$LOG" 2>&1 || true
cat > "${OUT_DIR}/MANIFEST.txt" <<EOF
Wave: 2 — ARCH_HEADS
Date: ${DAY}
Model: hgt L${HGT_LAYERS}
Heads: 2,4,8
Seeds: 20260721-20260723
CSV: ${DAY}_wave2_ARCH_HEADS_all_runs.csv
W&B: TaskA_ARCH_HEADS
EOF
echo "ARCH_HEADS finished $(date) → ${OUT_DIR}" | tee -a "$LOG"
