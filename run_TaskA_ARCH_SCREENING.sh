#!/usr/bin/env bash
# Wave 2 — architecture screening (matched encoder family).
# Saves logs + CSV exports under:
#   outputs/wave2_architecture_YYYY-MM-DD/
#
# Do not confuse with wave1 (BAZA / E1–E4) in:
#   outputs/wave1_baselines_ablations/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="$(date +%Y-%m-%d)"
OUT_DIR="outputs/wave2_architecture_${DAY}"
mkdir -p "$OUT_DIR" "$OUT_DIR/by_model" "$OUT_DIR/hydra_run_dirs"
LOG="${OUT_DIR}/${DAY}_wave2_ARCH_SCREENING_run.log"

# Snapshot wave1 labels once so folders stay unambiguous.
python scripts/export_wave2_architecture_results.py --day "$DAY" --wave1-snapshot

SEEDS=(20260721 20260722 20260723)
MODELS=(
  TaskA_hetero_sage_matched
  TaskA_hetero_gatv2
  TaskA_hgt
)

{
  echo "============================================================"
  echo "WAVE 2 — ARCHITECTURE SCREENING"
  echo "Date: ${DAY}"
  echo "Out:  ${OUT_DIR}"
  echo "Models: ${MODELS[*]}"
  echo "Seeds:  ${SEEDS[*]}"
  echo "Frozen: clean | empirical | candidate_seed=20260722 | no HCR"
  echo "Question: attention (GATv2/HGT) vs matched HeteroSAGE?"
  echo "============================================================"
} | tee "$LOG"

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    short="${model#TaskA_}"
    echo "------------------------------------------------------------" | tee -a "$LOG"
    echo "START model=${short} seed=${seed} $(date '+%H:%M:%S')" | tee -a "$LOG"

    python src/train_taskA.py \
      model="$model" \
      model.num_layers=2 \
      data.dataset.scenario=clean \
      data.feature_ablation_profile=empirical \
      data.candidate_seed=20260722 \
      training.seed="$seed" \
      experiment.wave=ARCH_SCREENING \
      experiment.intervention="arch_${short}" \
      experiment.hcr_variant=none \
      wandb.enabled=true \
      wandb.group=TaskA_ARCH_SCREENING \
      wandb.job_type=architecture_screening \
      "wandb.tags=[TaskA,WAVE2,ARCH_SCREENING,v3,clean,no_hcr,${short}]" \
      2>&1 | tee -a "$LOG"

    echo "DONE  model=${short} seed=${seed} $(date '+%H:%M:%S')" | tee -a "$LOG"

    # Refresh dated CSV exports after each run (safe to re-run).
    python scripts/export_wave2_architecture_results.py --day "$DAY" | tee -a "$LOG"
  done
done

python scripts/export_wave2_architecture_results.py --day "$DAY" | tee -a "$LOG"

cat > "${OUT_DIR}/MANIFEST.txt" <<EOF
Wave: 2 — architecture screening (matched encoder family)
Date folder: ${DAY}
Models: hetero_sage_matched, hetero_gatv2, hgt
Seeds: 20260721 20260722 20260723
Log: ${DAY}_wave2_ARCH_SCREENING_run.log
CSV: ${DAY}_wave2_ARCH_SCREENING_all_models.csv
Summary: ${DAY}_wave2_ARCH_SCREENING_summary_by_model.csv
Per-model: by_model/${DAY}_wave2_<model>_seed-runs.csv

NOT wave1. Wave1 baselines/ablations live in:
  outputs/wave1_baselines_ablations/

W&B group: TaskA_ARCH_SCREENING
EOF

echo "Wave2 finished. Results in: ${OUT_DIR}" | tee -a "$LOG"
