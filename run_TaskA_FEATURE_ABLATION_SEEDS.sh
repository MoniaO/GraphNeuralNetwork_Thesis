#!/usr/bin/env bash
# Task A — feature ablation multi-seed confirmation
#
# 2 models × 3 profiles × 5 seeds = 30 runs
# Single-seed ablation was diagnostic; this estimates Δ mean ± SD.
#
# Depths frozen from TaskA_BAZA / clean:
#   HeteroSAGE → L2
#   R-GCN      → L1
#
# candidate_seed and feature_ablation_seed stay frozen.
# Only training.seed varies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Brak .venv w: $PWD"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

FEATURE_PROFILES="empirical,topology_only,empirical_shuffled"
TRAIN_SEEDS="20260721,20260722,20260723,20260724,20260725"

echo "=========================================================="
echo "Task A feature ablation — multi-seed"
echo "Scenario: clean"
echo "Candidate seed: 20260722 (frozen)"
echo "Feature ablation seed: 20260722 (frozen)"
echo "Training seeds: ${TRAIN_SEEDS}"
echo "Profiles: ${FEATURE_PROFILES}"
echo "Matrix: 2 × 3 × 5 = 30 runs"
echo "=========================================================="

echo "Running audit..."
python scripts/audit_taskA_feature_ablation.py

echo "HeteroSAGE L2 × 3 profiles × 5 seeds..."
python src/train_taskA.py -m \
  model=TaskA_hetero_sage \
  model.num_layers=2 \
  data.dataset.scenario=clean \
  data.feature_ablation_profile="${FEATURE_PROFILES}" \
  data.feature_ablation_seed=20260722 \
  data.candidate_seed=20260722 \
  training.seed="${TRAIN_SEEDS}" \
  experiment.wave=FEATURE_ABLATION_SEEDS \
  experiment.hcr_variant=none \
  wandb.enabled=true \
  wandb.group=TaskA_FEATURE_ABLATION_SEEDS \
  wandb.job_type=feature_ablation_seeds \
  'wandb.tags=[TaskA,FEATURE_ABLATION,SEEDS,v3,clean,no_hcr]'

echo "R-GCN L1 × 3 profiles × 5 seeds..."
python src/train_taskA.py -m \
  model=TaskA_rgcn \
  model.num_layers=1 \
  data.dataset.scenario=clean \
  data.feature_ablation_profile="${FEATURE_PROFILES}" \
  data.feature_ablation_seed=20260722 \
  data.candidate_seed=20260722 \
  training.seed="${TRAIN_SEEDS}" \
  experiment.wave=FEATURE_ABLATION_SEEDS \
  experiment.hcr_variant=none \
  wandb.enabled=true \
  wandb.group=TaskA_FEATURE_ABLATION_SEEDS \
  wandb.job_type=feature_ablation_seeds \
  'wandb.tags=[TaskA,FEATURE_ABLATION,SEEDS,v3,clean,no_hcr]'

echo "TaskA_FEATURE_ABLATION_SEEDS finished (30 runs)."
