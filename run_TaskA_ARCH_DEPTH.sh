#!/usr/bin/env bash
# Wave 2 / DEPTH — robust runner (survives export glitches; skips finished runs).
# 2 models × 3 depths × 3 seeds = 18
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="$(date +%Y-%m-%d)"
OUT_DIR="outputs/wave2_architecture_${DAY}_DEPTH"
mkdir -p "$OUT_DIR" "$OUT_DIR/by_model"
LOG="${OUT_DIR}/${DAY}_wave2_ARCH_DEPTH_VISIBLE.log"
CSV="${OUT_DIR}/${DAY}_wave2_ARCH_DEPTH_all_runs.csv"

SEEDS=(20260721 20260722 20260723)
LAYERS=(1 2 3)
MODELS=(TaskA_hgt TaskA_hetero_sage_matched)

already_done() {
  local model="$1" layers="$2" seed="$3"
  python - "$CSV" "$model" "$layers" "$seed" <<'PY'
import sys
from pathlib import Path
import pandas as pd
csv, model, layers, seed = sys.argv[1:5]
p = Path(csv)
if not p.exists():
    raise SystemExit(1)
df = pd.read_csv(p)
m = (
    (df["model"].astype(str) == model)
    & (df["num_layers"].astype(int) == int(layers))
    & (df["training_seed"].astype(int) == int(seed))
    & df["valid_auprc"].notna()
)
raise SystemExit(0 if m.any() else 1)
PY
}

{
  echo "============================================================"
  echo "WAVE 2 — ARCH DEPTH (robust)"
  echo "Date: ${DAY}"
  echo "Out:  ${OUT_DIR}"
  echo "Log:  ${LOG}"
  echo "Total planned: 18 | device=cpu | skip finished"
  echo "============================================================"
} | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for layers in "${LAYERS[@]}"; do
      short="${model#TaskA_}"
      label="${short}_L${layers}"

      if already_done "$short" "$layers" "$seed"; then
        echo "SKIP ${label} seed=${seed} (already in CSV)" | tee -a "$LOG"
        continue
      fi

      echo "===== START ${label} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"

      set +e
      python src/train_taskA.py \
        model="$model" \
        model.num_layers="$layers" \
        data.dataset.scenario=clean \
        data.feature_ablation_profile=empirical \
        data.candidate_seed=20260722 \
        training.seed="$seed" \
        training.device=cpu \
        experiment.wave=ARCH_DEPTH \
        experiment.intervention="depth_${label}" \
        experiment.hcr_variant=none \
        wandb.enabled=true \
        wandb.group=TaskA_ARCH_DEPTH \
        wandb.job_type=architecture_depth \
        "wandb.tags=[TaskA,WAVE2,ARCH_DEPTH,v3,clean,no_hcr,${short},L${layers}]" \
        >>"$LOG" 2>&1
      rc=$?

      echo "===== DONE ${label} seed=${seed} rc=${rc} $(date '+%H:%M:%S') =====" | tee -a "$LOG"

      python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_DEPTH >>"$LOG" 2>&1 || true
    done
  done
done

python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_DEPTH >>"$LOG" 2>&1 || true
echo "ARCH_DEPTH finished $(date)" | tee -a "$LOG"
