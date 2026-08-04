#!/usr/bin/env bash
# E4A — Relation ablation (message-passing only), HeteroSAGE L2 × 5 seeds
#
# Removes one forward relation (+ its rev_*) from G-TRAIN.
# Candidates / labels stay fixed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

SEEDS=(20260721 20260722 20260723 20260724 20260725)
# Top forward relations by message-edge support on clean G-TRAIN
RELATIONS=(
  none
  risk_modifier
  interaction_amplify
  drug_to_load
  drug_to_burden
  drug_to_mechanism
)

echo "E4A relation ablation: HeteroSAGE L2 × ${#SEEDS[@]} seeds × ${#RELATIONS[@]} interventions"

for seed in "${SEEDS[@]}"; do
  for relation in "${RELATIONS[@]}"; do
    if [[ "$relation" == "none" ]]; then
      enabled=false
      removed='[]'
      label=full_graph
    else
      enabled=true
      removed="[${relation}]"
      label="remove_${relation}"
    fi

    echo "------------------------------------------------------------"
    echo "seed=${seed} intervention=${label}"

    python src/train_taskA.py \
      model=TaskA_hetero_sage \
      model.num_layers=2 \
      data.dataset.scenario=clean \
      data.feature_ablation_profile=empirical \
      data.candidate_seed=20260722 \
      data.feature_ablation_seed=20260722 \
      training.seed="$seed" \
      experiment.wave=RELATION_ABLATION \
      experiment.intervention="$label" \
      experiment.relation_ablation.enabled="$enabled" \
      "experiment.relation_ablation.removed_relations=$removed" \
      wandb.enabled=true \
      wandb.group=TaskA_RELATION_ABLATION \
      wandb.job_type=relation_ablation \
      'wandb.tags=[TaskA,E4,RELATION_ABLATION,v3,clean]'
  done
done

echo "TaskA_RELATION_ABLATION finished."
