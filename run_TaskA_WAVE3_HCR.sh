#!/usr/bin/env bash
# Wave 3 / HCR stage 1 — frozen HGT L1 H8 + decoder-side HCR variants.
# Matrix: {binary_compact, classical_binary, binary_minimal} × 3 seeds = 9 runs
# HCR-0 baseline = existing FINAL HGT L1 H8 (do not retrain here).
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
LOG="${OUT_DIR}/${DAY}_wave3_HCR_VISIBLE.log"
CSV="${OUT_DIR}/${DAY}_wave3_HCR_all_runs.csv"

SEEDS=(20260721 20260722 20260723)
VARIANTS=(binary_compact classical_binary binary_minimal)

already_done() {
  local variant="$1" seed="$2"
  python - "$CSV" "$variant" "$seed" <<'PY'
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

export_local() {
  python - "$OUT_DIR" "$DAY" <<'PY'
from __future__ import annotations
import json, re, sys
from pathlib import Path
import pandas as pd

out_dir = Path(sys.argv[1])
day = sys.argv[2]
WANDB = Path('wandb')
rows = []
for run_dir in sorted(WANDB.glob('run-*')):
    sp = run_dir / 'files' / 'wandb-summary.json'
    if not sp.exists():
        continue
    summary = json.loads(sp.read_text())
    blob = ''
    cfg = run_dir / 'files' / 'config.yaml'
    if cfg.exists():
        blob += cfg.read_text(errors='ignore')
    meta = run_dir / 'files' / 'wandb-metadata.json'
    if meta.exists():
        blob += meta.read_text(errors='ignore')
    blob += json.dumps(summary)
    if 'WAVE3_HCR' not in blob and summary.get('experiment_name') != 'WAVE3_HCR':
        continue
    seed = summary.get('training_seed')
    if seed is None:
        m = re.search(r'training\\.seed[=:]\\s*(\\d+)', blob)
        seed = int(m.group(1)) if m else None
    variant = summary.get('hcr_variant') or summary.get('intervention')
    if variant is None:
        m = re.search(r'hcr=([a-z_]+)', blob)
        variant = m.group(1) if m else None
    rows.append({
        'wave': 'wave3_hcr',
        'stage': 'WAVE3_HCR',
        'model': 'hgt',
        'num_layers': 1,
        'heads': 8,
        'hcr_variant': variant,
        'training_seed': int(seed) if seed is not None else None,
        'valid_auprc': summary.get('valid_auprc', summary.get('best_valid_AUPRC', summary.get('best_valid_metric'))),
        'valid_auroc': summary.get('valid_auc', summary.get('valid_auroc')),
        'valid_brier': summary.get('valid_brier'),
        'test_auprc': summary.get('test_auprc'),
        'test_auroc': summary.get('test_auc', summary.get('test_auroc')),
        'test_brier': summary.get('test_brier'),
        'parameter_count_trainable': summary.get('parameter_count_trainable'),
        'best_epoch': summary.get('best_epoch'),
        'run_dir': run_dir.name,
        'mtime': run_dir.stat().st_mtime,
    })
df = pd.DataFrame(rows).dropna(subset=['training_seed', 'valid_auprc', 'hcr_variant'])
if df.empty:
    print('No WAVE3_HCR runs exported yet')
    raise SystemExit(0)
df = df.sort_values('mtime').drop_duplicates(['hcr_variant', 'training_seed'], keep='first')
csv = out_dir / f'{day}_wave3_HCR_all_runs.csv'
df.to_csv(csv, index=False)
summary = (
    df.groupby('hcr_variant', dropna=False)
    .agg(
        n_seeds=('training_seed', 'nunique'),
        mean_valid_auprc=('valid_auprc', 'mean'),
        std_valid_auprc=('valid_auprc', 'std'),
        mean_test_auprc=('test_auprc', 'mean'),
        std_test_auprc=('test_auprc', 'std'),
    )
    .reset_index()
    .sort_values('mean_valid_auprc', ascending=False)
)
summary.to_csv(out_dir / f'{day}_wave3_HCR_summary.csv', index=False)
print(summary.to_string(index=False))
print('Wrote', csv)
PY
}

{
  echo "============================================================"
  echo "WAVE 3 — HCR stage 1 (frozen HGT L1 H8)"
  echo "Date: ${DAY}"
  echo "Variants: ${VARIANTS[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "device=cpu | skip finished"
  echo "============================================================"
} | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    label="hgt_hcr_${variant}"
    if already_done "$variant" "$seed"; then
      echo "SKIP ${label} seed=${seed}" | tee -a "$LOG"
      continue
    fi
    echo "===== START ${label} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    python src/train_taskA.py \
      model=TaskA_hgt_hcr \
      "hcr=${variant}" \
      model.num_layers=1 \
      model.hidden_dim=64 \
      model.hidden_channels=64 \
      model.heads=8 \
      model.dropout=0.2 \
      data.dataset.scenario=clean \
      data.feature_ablation_profile=empirical \
      data.feature_ablation_seed=20260722 \
      data.candidate_seed=20260722 \
      training.seed="$seed" \
      training.device=cpu \
      experiment.wave=WAVE3_HCR \
      "experiment.intervention=hgt_hcr_${variant}" \
      "experiment.hcr_variant=${variant}" \
      wandb.enabled=true \
      wandb.group=TaskA_WAVE3_HCR \
      wandb.job_type=hcr_stage1 \
      "wandb.tags=[TaskA,WAVE3,HCR,v3,clean,hgt,L1,H8,${variant}]" \
      >>"$LOG" 2>&1
    rc=$?
    echo "===== DONE ${label} seed=${seed} rc=${rc} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    export_local >>"$LOG" 2>&1 || true
  done
done

export_local | tee -a "$LOG"
cat > "${OUT_DIR}/MANIFEST.txt" <<EOF
Wave: 3 — HCR stage 1
Date: ${DAY}
Frozen encoder: HGT L1 heads=8 hidden=64
Variants: binary_compact, classical_binary, binary_minimal
Seeds: 20260721-20260723
Baseline HCR-0: outputs/wave2_architecture_*_FINAL (HGT L1 H8)
CSV: ${DAY}_wave3_HCR_all_runs.csv
EOF
echo "WAVE3_HCR finished $(date) → ${OUT_DIR}" | tee -a "$LOG"
