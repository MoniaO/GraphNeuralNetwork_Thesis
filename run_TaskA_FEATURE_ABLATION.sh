#!/usr/bin/env bash
# Task A — feature ablation on clean (6 runs)
#
# Interpretation:
#   empirical > topology_only  → patient features add information
#   empirical ≈ topology_only  → model relies mainly on graph structure
#   empirical ≈ shuffled       → concrete node↔feature assignment unused
#
# Depths frozen from TaskA_BAZA clean selection:
#   HeteroSAGE → L2
#   R-GCN      → L1
#
# Do NOT run this before: python scripts/audit_taskA_feature_ablation.py
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

echo "=========================================================="
echo "Task A feature ablation"
echo "Scenario: clean"
echo "Candidate seed: 20260722"
echo "Feature ablation seed: 20260722"
echo "Training seed: 20260721"
echo "Profiles: ${FEATURE_PROFILES}"
echo "Models: HeteroSAGE L2, R-GCN L1 (frozen from BAZA/clean)"
echo "=========================================================="

echo "Running audit..."
python scripts/audit_taskA_feature_ablation.py

echo "Running HeteroSAGE L2 × 3 profiles..."
python src/train_taskA.py -m \
  model=TaskA_hetero_sage \
  model.num_layers=2 \
  data.dataset.scenario=clean \
  data.feature_ablation_profile="${FEATURE_PROFILES}" \
  data.feature_ablation_seed=20260722 \
  data.candidate_seed=20260722 \
  training.seed=20260721 \
  experiment.wave=FEATURE_ABLATION \
  experiment.hcr_variant=none \
  wandb.enabled=true \
  wandb.group=TaskA_FEATURE_ABLATION \
  wandb.job_type=feature_ablation \
  'wandb.tags=[TaskA,FEATURE_ABLATION,v3,clean,no_hcr]'

echo "Running R-GCN L1 × 3 profiles..."
python src/train_taskA.py -m \
  model=TaskA_rgcn \
  model.num_layers=1 \
  data.dataset.scenario=clean \
  data.feature_ablation_profile="${FEATURE_PROFILES}" \
  data.feature_ablation_seed=20260722 \
  data.candidate_seed=20260722 \
  training.seed=20260721 \
  experiment.wave=FEATURE_ABLATION \
  experiment.hcr_variant=none \
  wandb.enabled=true \
  wandb.group=TaskA_FEATURE_ABLATION \
  wandb.job_type=feature_ablation \
  'wandb.tags=[TaskA,FEATURE_ABLATION,v3,clean,no_hcr]'

echo "TaskA_FEATURE_ABLATION finished (6 runs)."
