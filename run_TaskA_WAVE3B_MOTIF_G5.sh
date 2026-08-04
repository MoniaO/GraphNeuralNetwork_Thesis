#!/usr/bin/env bash
# Wave 3B G5 only — hcr3_random_context × 3 seeds (closes motif matrix)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="${WAVE3B_DAY:-2026-07-31}"
OUT_DIR="outputs/wave3b_motif_${DAY}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/${DAY}_wave3b_MOTIF_G5.log"
SEEDS=(20260721 20260722 20260723)
VARIANT=hcr3_random_context

append_row() {
  local variant="$1" seed="$2" ckpt="$3"
  python - "$OUT_DIR" "$DAY" "$variant" "$seed" "$ckpt" <<'PY'
import json, sys, os
from pathlib import Path
import pandas as pd
import torch
out_dir, day, variant, seed, ckpt = sys.argv[1:6]
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
from subprocess import check_output
check_output([
    sys.executable, "scripts/eval_wave3b_motif_completion.py",
    "--checkpoint", ckpt, "--hcr", variant, "--seed", seed,
    "--out", str(Path(out_dir) / f"motif_metrics_{variant}_{seed}.json"),
], env={**os.environ, "PYTHONPATH": "src:."})
metrics = json.loads(Path(out_dir, f"motif_metrics_{variant}_{seed}.json").read_text())
row["n_motif_edges_found"] = metrics.get("n_motif_edges_found")
row["n_motif_positives"] = metrics.get("n_motif_positives")
row["valid_auprc_gate_pool"] = metrics.get("auprc_motif_heldout_vs_valid_neg")
row["valid_auprc_motif_only"] = metrics.get("auprc_motif_heldout_vs_valid_neg")
row["mean_prob_motif"] = metrics.get("mean_prob_motif")
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
  echo "WAVE 3B — G5 ${VARIANT}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Started: $(date)"
  echo "============================================================"
} | tee "$LOG"

for seed in "${SEEDS[@]}"; do
  echo "===== START ${VARIANT} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python src/train_taskA.py \
    model=TaskA_hgt_hcr "hcr=${VARIANT}" \
    model.num_layers=1 model.hidden_dim=64 model.hidden_channels=64 \
    model.heads=8 model.dropout=0.2 \
    data.dataset.scenario=clean \
    data.feature_ablation_profile=empirical \
    data.candidate_seed=20260722 \
    training.seed="$seed" training.device=cpu \
    experiment.wave=WAVE3B_MOTIF \
    experiment.motif_completion.enabled=true \
    experiment.motif_completion.hide=parent_a \
    "experiment.intervention=motif_hide_parent_a__${VARIANT}" \
    "experiment.hcr_variant=${VARIANT}" \
    wandb.enabled=false \
    >>"$LOG" 2>&1
  rc=$?
  echo "===== DONE ${VARIANT} seed=${seed} rc=${rc} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  CKPT="$(python - <<PY
import json
from pathlib import Path
best=None; best_m=0
for p in Path("wandb").glob("run-*/files/wandb-summary.json"):
    s=json.loads(p.read_text())
    if s.get("experiment_name")!="WAVE3B_MOTIF":
        continue
    if int(s.get("training_seed") or -1)!=${seed}:
        continue
    if s.get("hcr_variant")!="${VARIANT}" and "${VARIANT}" not in str(s.get("intervention") or ""):
        continue
    ck=s.get("checkpoint_path")
    if ck and Path(ck).exists() and p.stat().st_mtime>=best_m:
        best, best_m = ck, p.stat().st_mtime
print(best or "")
PY
)"
  if [[ -n "$CKPT" ]]; then
    append_row "$VARIANT" "$seed" "$CKPT" >>"$LOG" 2>&1 || echo "WARN: append_row failed" | tee -a "$LOG"
  else
    echo "WARN: no checkpoint for ${VARIANT} seed=${seed}" | tee -a "$LOG"
  fi
done

python - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
import pandas as pd
out = Path("outputs/wave3b_motif_2026-07-31")
rows = []
for p in sorted(out.glob("motif_metrics_*.json")):
    name = p.stem.replace("motif_metrics_", "")
    *vp, seed = name.rsplit("_", 1)
    d = json.loads(p.read_text())
    rows.append({
        "variant": "_".join(vp),
        "seed": int(seed),
        "auprc": d["auprc_motif_heldout_vs_valid_neg"],
        "mean_prob": d["mean_prob_motif"],
    })
df = pd.DataFrame(rows)
print("\n=== Motif AUPRC by variant ===")
print(df.groupby("variant")["auprc"].agg(["mean", "std", "count"]).round(4).to_string())
means = df.groupby("variant")["auprc"].mean()
if "hcr3_random_context" in means.index and "hcr3_full" in means.index:
    print(f"\nΔ G2−G5 (full − random_context): {means['hcr3_full']-means['hcr3_random_context']:+.4f}")
if "hcr3_shuffled" in means.index and "hcr3_full" in means.index:
    print(f"Δ G2−G4 (full − shuffled): {means['hcr3_full']-means['hcr3_shuffled']:+.4f}")
PY

# update PROTOCOL
python - <<'PY'
from pathlib import Path
p = Path("outputs/wave3b_motif_2026-07-31/PROTOCOL.txt")
txt = p.read_text() if p.exists() else ""
if "G5 completed" not in txt:
    p.write_text(txt.rstrip() + "\n\nG5 completed: hcr3_random_context × seeds 20260721–23\n")
PY

echo "G5 FINISHED $(date) → ${OUT_DIR}" | tee -a "$LOG"
