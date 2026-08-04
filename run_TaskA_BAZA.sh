#!/usr/bin/env bash
# Task A — Wave 1 (BAZA): 6 scenarios × 2 models × 4 depths = 48 runs
# No HCR. Candidate graph/splits frozen via data.candidate_seed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
# Back-compat if any leftover config still reads PHARMA_DATA_ROOT
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

echo "GSN_PROJECT_ROOT=$GSN_PROJECT_ROOT"
echo "PHARMA_DATA_ROOT=$PHARMA_DATA_ROOT"
echo "Starting TaskA_BAZA multirun (48 jobs, sequential)..."

python src/train_taskA.py -m \
  model=TaskA_hetero_sage,TaskA_rgcn \
  data.dataset.scenario=clean,hidden_confounder,selection_bias,no_overlap,noisy_documentation,multihospital \
  model.num_layers=1,2,3,4 \
  data.candidate_seed=20260722 \
  training.seed=20260721 \
  experiment.wave=BAZA \
  experiment.hcr_variant=none \
  wandb.enabled=true \
  wandb.group=TaskA_BAZA \
  wandb.job_type=training \
  'wandb.tags=[TaskA,BAZA,v3,no_hcr]'

echo "TaskA_BAZA finished."
