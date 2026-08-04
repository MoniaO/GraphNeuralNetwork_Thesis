#!/usr/bin/env bash
# Wave 3B / A1 — dual-gate motif completion (hide parent_a→gate from G_train).
# Variants: none | binary_compact | hcr3_full | hcr3_without_a111 | hcr3_shuffled
# Seeds: 3
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="$(date +%Y-%m-%d)"
OUT_DIR="outputs/wave3b_motif_${DAY}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/${DAY}_wave3b_MOTIF_VISIBLE.log"
CSV="${OUT_DIR}/${DAY}_wave3b_MOTIF_all_runs.csv"

SEEDS=(20260721 20260722 20260723)
VARIANTS=(none binary_compact hcr3_full hcr3_without_a111 hcr3_shuffled)

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
m = (df["hcr_variant"].astype(str) == variant) & (df["training_seed"].astype(int) == int(seed)) & df["valid_auprc"].notna()
raise SystemExit(0 if m.any() else 1)
PY
}

append_row() {
  local variant="$1" seed="$2" ckpt="$3"
  python - "$OUT_DIR" "$DAY" "$variant" "$seed" "$ckpt" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
out_dir, day, variant, seed, ckpt = sys.argv[1:6]
# read wandb summary near ckpt parent is unreliable; parse train log metrics from checkpoint
import torch
payload = torch.load(ckpt, map_location="cpu", weights_only=False)
final_valid = payload.get("final_valid_metrics") or {}
final_test = payload.get("final_test_metrics") or {}
row = {
    "wave": "wave3b_motif",
    "hcr_variant": variant,
    "training_seed": int(seed),
    "valid_auprc": final_valid.get("auprc", payload.get("best_valid_metric")),
    "valid_brier": final_valid.get("brier"),
    "test_auprc": final_test.get("auprc"),
    "test_brier": final_test.get("brier"),
    "best_epoch": payload.get("best_epoch"),
    "checkpoint_path": ckpt,
}
# gate metrics
from subprocess import check_output
raw = check_output([
    sys.executable, "scripts/eval_wave3b_motif_completion.py",
    "--checkpoint", ckpt, "--hcr", variant, "--seed", seed,
    "--out", str(Path(out_dir) / f"motif_metrics_{variant}_{seed}.json"),
], env={**__import__("os").environ, "PYTHONPATH": "src:."})
metrics = json.loads(Path(out_dir, f"motif_metrics_{variant}_{seed}.json").read_text())
row["n_motif_edges_found"] = metrics.get("n_motif_edges_found")
row["n_motif_positives"] = metrics.get("n_motif_positives")
row["valid_auprc_gate_pool"] = metrics.get("auprc_motif_heldout_vs_valid_neg")
row["valid_auprc_motif_only"] = metrics.get("auprc_motif_heldout_vs_valid_neg")
row["mean_prob_motif"] = metrics.get("mean_prob_motif")
# keep split overall from checkpoint
for m in metrics.get("splits", []):
    if m["split"] == "valid":
        row["valid_auprc_reeval"] = m["auprc_all"]
    if m["split"] == "test":
        row["test_auprc_reeval"] = m["auprc_all"]
csv = Path(out_dir) / f"{day}_wave3b_MOTIF_all_runs.csv"
df = pd.DataFrame([row])
if csv.exists():
    old = pd.read_csv(csv)
    df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(["hcr_variant", "training_seed"], keep="last")
df.to_csv(csv, index=False)
print(df.tail(1).to_string(index=False))
PY
}

{
  echo "============================================================"
  echo "WAVE 3B — MOTIF COMPLETION (hide parent_a→gate)"
  echo "Variants: ${VARIANTS[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "============================================================"
} | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    if already_done "$variant" "$seed"; then
      echo "SKIP ${variant} seed=${seed}" | tee -a "$LOG"
      continue
    fi
    echo "===== START ${variant} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    if [[ "$variant" == "none" ]]; then
      MODEL_ARGS=(model=TaskA_hgt hcr=none)
    else
      MODEL_ARGS=(model=TaskA_hgt_hcr "hcr=${variant}")
    fi
    python src/train_taskA.py \
      "${MODEL_ARGS[@]}" \
      model.num_layers=1 \
      model.hidden_dim=64 \
      model.hidden_channels=64 \
      model.heads=8 \
      model.dropout=0.2 \
      data.dataset.scenario=clean \
      data.feature_ablation_profile=empirical \
      data.candidate_seed=20260722 \
      training.seed="$seed" \
      training.device=cpu \
      experiment.wave=WAVE3B_MOTIF \
      experiment.motif_completion.enabled=true \
      experiment.motif_completion.hide=parent_a \
      "experiment.intervention=motif_hide_parent_a__${variant}" \
      "experiment.hcr_variant=${variant}" \
      wandb.enabled=true \
      wandb.group=TaskA_WAVE3B_MOTIF \
      wandb.job_type=motif_completion \
      "wandb.tags=[TaskA,WAVE3B,MOTIF,hgt,L1,H8,${variant}]" \
      >>"$LOG" 2>&1
    rc=$?
    echo "===== DONE ${variant} seed=${seed} rc=${rc} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    # recover checkpoint path from latest matching wandb summary
    CKPT="$(
      VARIANT="$variant" SEED="$seed" python - <<'PY'
import json
import os
from pathlib import Path

variant = os.environ["VARIANT"]
seed = int(os.environ["SEED"])
best = None
best_m = 0
for p in Path("wandb").glob("run-*/files/wandb-summary.json"):
    s = json.loads(p.read_text())
    if s.get("experiment_name") != "WAVE3B_MOTIF":
        continue
    if int(s.get("training_seed") or -1) != seed:
        continue
    interv = str(s.get("intervention") or "")
    hcr_variant = s.get("hcr_variant")
    if variant not in interv and hcr_variant != variant:
        if variant == "none" and (
            hcr_variant in (None, "none") or "none" in interv
        ):
            pass
        else:
            continue
    ck = s.get("checkpoint_path")
    if ck and Path(ck).exists() and p.stat().st_mtime >= best_m:
        best, best_m = ck, p.stat().st_mtime
print(best or "")
PY
    )"
    if [[ -n "${CKPT}" ]]; then
      append_row "${variant}" "${seed}" "${CKPT}" >>"$LOG" 2>&1 || true
    else
      echo "WARN: no checkpoint for ${variant} seed=${seed}" | tee -a "$LOG"
    fi
  done
done

python - <<PY | tee -a "$LOG"
import pandas as pd
from pathlib import Path
day="${DAY}"
p=Path(f"outputs/wave3b_motif_{day}/{day}_wave3b_MOTIF_all_runs.csv")
if not p.exists():
    print("No results CSV yet"); raise SystemExit(0)
df=pd.read_csv(p)
summary=(df.groupby("hcr_variant")
 .agg(n=("training_seed","nunique"),
      mean_valid=("valid_auprc","mean"),
      mean_valid_gate=("valid_auprc_gate_pool","mean"),
      mean_test=("test_auprc","mean"),
      mean_test_gate=("test_auprc_gate_pool","mean"))
 .reset_index()
 .sort_values("mean_valid_gate", ascending=False))
summary.to_csv(f"outputs/wave3b_motif_{day}/{day}_wave3b_MOTIF_summary.csv", index=False)
print(summary.to_string(index=False))
Path(f"outputs/wave3b_motif_{day}/MANIFEST.txt").write_text(
    "Wave 3B — motif completion A1\\n"
    "Hide: parent_a→gate for each binary dual gate (10 edges) from G_train MP\\n"
    "Keep: parent_b→gate in message passing\\n"
    "Primary metric: valid_auprc_gate_pool\\n"
    f"Results: {p}\\n"
)
PY

echo "WAVE3B MOTIF finished $(date) → ${OUT_DIR}" | tee -a "$LOG"
