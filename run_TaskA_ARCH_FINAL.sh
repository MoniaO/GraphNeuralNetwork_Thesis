#!/usr/bin/env bash
# Wave 2 / FINAL — locked HGT vs HeteroSAGE matched on 5 seeds.
#   HGT_LAYERS=1 HGT_HEADS=4 SAGE_LAYERS=1 ./run_TaskA_ARCH_FINAL.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="$(date +%Y-%m-%d)"
OUT_DIR="outputs/wave2_architecture_${DAY}_FINAL"
mkdir -p "$OUT_DIR" "$OUT_DIR/by_model"
LOG="${OUT_DIR}/${DAY}_wave2_ARCH_FINAL_VISIBLE.log"
CSV="${OUT_DIR}/${DAY}_wave2_ARCH_FINAL_all_runs.csv"

HGT_LAYERS="${HGT_LAYERS:-1}"
HGT_HEADS="${HGT_HEADS:-4}"
SAGE_LAYERS="${SAGE_LAYERS:-1}"
SEEDS=(20260721 20260722 20260723 20260724 20260725)

already_done() {
  local model="$1" layers="$2" heads="$3" seed="$4"
  python - "$CSV" "$model" "$layers" "$heads" "$seed" <<'PY'
import sys
from pathlib import Path
import pandas as pd
csv, model, layers, heads, seed = sys.argv[1:6]
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
if model == "hgt" and "heads" in df.columns:
    m = m & (df["heads"].fillna(-1).astype(int) == int(heads))
raise SystemExit(0 if m.any() else 1)
PY
}

{
  echo "============================================================"
  echo "WAVE 2 — ARCH FINAL (robust)"
  echo "Date: ${DAY}"
  echo "HGT: L${HGT_LAYERS} heads=${HGT_HEADS}"
  echo "SAGE matched: L${SAGE_LAYERS}"
  echo "Seeds: ${SEEDS[*]}"
  echo "device=cpu | skip finished"
  echo "============================================================"
} | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
  label_hgt="hgt_L${HGT_LAYERS}_H${HGT_HEADS}"
  if already_done "hgt" "$HGT_LAYERS" "$HGT_HEADS" "$seed"; then
    echo "SKIP ${label_hgt} seed=${seed}" | tee -a "$LOG"
  else
    echo "===== START ${label_hgt} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    python src/train_taskA.py \
      model=TaskA_hgt \
      model.num_layers="$HGT_LAYERS" \
      model.heads="$HGT_HEADS" \
      data.dataset.scenario=clean \
      data.feature_ablation_profile=empirical \
      data.candidate_seed=20260722 \
      training.seed="$seed" \
      training.device=cpu \
      experiment.wave=ARCH_FINAL \
      experiment.intervention="final_${label_hgt}" \
      experiment.hcr_variant=none \
      wandb.enabled=true \
      wandb.group=TaskA_ARCH_FINAL \
      wandb.job_type=architecture_final \
      "wandb.tags=[TaskA,WAVE2,ARCH_FINAL,v3,clean,no_hcr,hgt]" \
      >>"$LOG" 2>&1
    echo "===== DONE ${label_hgt} seed=${seed} rc=$? $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_FINAL >>"$LOG" 2>&1 || true
  fi

  label_sage="hetero_sage_matched_L${SAGE_LAYERS}"
  if already_done "hetero_sage_matched" "$SAGE_LAYERS" "-1" "$seed"; then
    echo "SKIP ${label_sage} seed=${seed}" | tee -a "$LOG"
  else
    echo "===== START ${label_sage} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    python src/train_taskA.py \
      model=TaskA_hetero_sage_matched \
      model.num_layers="$SAGE_LAYERS" \
      data.dataset.scenario=clean \
      data.feature_ablation_profile=empirical \
      data.candidate_seed=20260722 \
      training.seed="$seed" \
      training.device=cpu \
      experiment.wave=ARCH_FINAL \
      experiment.intervention="final_${label_sage}" \
      experiment.hcr_variant=none \
      wandb.enabled=true \
      wandb.group=TaskA_ARCH_FINAL \
      wandb.job_type=architecture_final \
      "wandb.tags=[TaskA,WAVE2,ARCH_FINAL,v3,clean,no_hcr,hetero_sage_matched]" \
      >>"$LOG" 2>&1
    echo "===== DONE ${label_sage} seed=${seed} rc=$? $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_FINAL >>"$LOG" 2>&1 || true
  fi
done

python scripts/export_wave2_architecture_results.py --day "$DAY" --stage ARCH_FINAL >>"$LOG" 2>&1 || true
cat > "${OUT_DIR}/MANIFEST.txt" <<EOF
Wave: 2 — ARCH_FINAL
Date: ${DAY}
HGT: L${HGT_LAYERS} heads=${HGT_HEADS}
SAGE matched: L${SAGE_LAYERS}
Seeds: 20260721-20260725
CSV: ${DAY}_wave2_ARCH_FINAL_all_runs.csv
Summary: ${DAY}_wave2_ARCH_FINAL_summary.csv
Notebook: notebooks/TaskA_WAVE2_architecture.ipynb
W&B: TaskA_ARCH_FINAL
EOF
echo "ARCH_FINAL finished $(date) → ${OUT_DIR}" | tee -a "$LOG"
