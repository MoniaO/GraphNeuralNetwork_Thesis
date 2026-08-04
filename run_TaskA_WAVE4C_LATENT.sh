#!/usr/bin/env bash
# Wave 4C — Latent-gate HCR recovery (hide A→G; features use Y not G).
# Variants L0–L5 × seeds 20260721–23. No Wave 5 path analysis.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export GSN_PROJECT_ROOT="${GSN_PROJECT_ROOT:-$HOME/Desktop/GSN Graphs dysertation 2026}"
export PHARMA_DATA_ROOT="${PHARMA_DATA_ROOT:-$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3}"

DAY="$(date +%Y-%m-%d)"
OUT_DIR="outputs/wave4c_latent_${DAY}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/${DAY}_wave4c_LATENT_VISIBLE.log"
CSV="${OUT_DIR}/${DAY}_wave4c_LATENT_all_runs.csv"

SEEDS=(20260721 20260722 20260723)
# L0 none | L1 pairwise ABY | L2 hcr3_full(A,B,Y) | L3 no a111 | L4 shuffle B | L5 random Z
VARIANTS=(none latent_pairwise_aby hcr3_full hcr3_without_a111 hcr3_shuffled hcr3_random_context)

{
  echo "============================================================"
  echo "WAVE 4C — LATENT GATE RECOVERY"
  echo "Hide A→G from G_train; HCR uses (A,Y,B) never G"
  echo "Variants: ${VARIANTS[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Started: $(date)"
  echo "============================================================"
} | tee "$LOG"

for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "===== START ${variant} seed=${seed} $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    if [[ "$variant" == "none" ]]; then
      MODEL_ARGS=(model=TaskA_hgt hcr=none)
    else
      MODEL_ARGS=(model=TaskA_hgt_hcr "hcr=${variant}")
    fi
    # L2–L5: force latent outcome y-mode (configs may still say oracle_coparent).
    EXTRA_ARGS=()
    if [[ "$variant" == "hcr3_random_context" ]]; then
      EXTRA_ARGS+=(hcr.z_mode=random_matched)
    elif [[ "$variant" == hcr3_* || "$variant" == "latent_pairwise_aby" ]]; then
      EXTRA_ARGS+=(hcr.z_mode=oracle_outcome)
    fi

    python src/train_taskA.py \
      "${MODEL_ARGS[@]}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
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
      experiment.wave=WAVE4C_LATENT \
      experiment.motif_completion.enabled=true \
      experiment.motif_completion.hide=parent_a \
      experiment.latent_gate.enabled=true \
      "experiment.intervention=latent_gate__${variant}" \
      "experiment.hcr_variant=${variant}" \
      wandb.enabled=false \
      >>"$LOG" 2>&1
    rc=$?
    echo "===== DONE ${variant} seed=${seed} rc=${rc} $(date '+%H:%M:%S') =====" | tee -a "$LOG"

    CKPT="$(rg -n "Saved checkpoint:" "$LOG" | tail -1 | sed 's/.*Saved checkpoint:[[:space:]]*//')"
    if [[ -n "$CKPT" && -f "$CKPT" ]]; then
      python scripts/eval_wave4c_latent_gate.py \
        --checkpoint "$CKPT" --hcr "$variant" --seed "$seed" \
        --out "${OUT_DIR}/latent_metrics_${variant}_${seed}.json" \
        >>"$LOG" 2>&1 || echo "WARN: eval failed" | tee -a "$LOG"
      python - "$OUT_DIR" "$DAY" "$variant" "$seed" "$CKPT" <<'PY' >>"$LOG" 2>&1 || true
import json, sys
from pathlib import Path
import pandas as pd
import torch
out_dir, day, variant, seed, ckpt = sys.argv[1:6]
payload = torch.load(ckpt, map_location="cpu", weights_only=False)
fv = payload.get("final_valid_metrics") or {}
ft = payload.get("final_test_metrics") or {}
row = {
    "wave": "wave4c_latent",
    "hcr_variant": variant,
    "training_seed": int(seed),
    "valid_auprc": fv.get("auprc", payload.get("best_valid_metric")),
    "valid_brier": fv.get("brier"),
    "test_auprc": ft.get("auprc"),
    "test_brier": ft.get("brier"),
    "best_epoch": payload.get("best_epoch"),
    "checkpoint_path": ckpt,
}
m = json.loads(Path(out_dir, f"latent_metrics_{variant}_{seed}.json").read_text())
row["auprc_motif"] = m.get("auprc_motif_heldout_vs_valid_neg")
row["mean_prob_motif"] = m.get("mean_prob_motif")
row["n_motif_positives"] = m.get("n_motif_positives")
csv = Path(out_dir) / f"{day}_wave4c_LATENT_all_runs.csv"
df = pd.DataFrame([row])
if csv.exists():
    old = pd.read_csv(csv)
    df = pd.concat([old, df], ignore_index=True).drop_duplicates(
        ["hcr_variant", "training_seed"], keep="last"
    )
df.to_csv(csv, index=False)
print(df.tail(1).to_string(index=False))
PY
    else
      echo "WARN: no checkpoint for ${variant} seed=${seed}" | tee -a "$LOG"
    fi
  done
done

python - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
import pandas as pd
import numpy as np

out = Path("outputs")
# pick latest wave4c dir
cands = sorted(out.glob("wave4c_latent_*"))
if not cands:
    print("No wave4c dir"); raise SystemExit(0)
d = cands[-1]
rows = []
for p in sorted(d.glob("latent_metrics_*.json")):
    name = p.stem.replace("latent_metrics_", "")
    *vp, seed = name.rsplit("_", 1)
    m = json.loads(p.read_text())
    rows.append({"variant": "_".join(vp), "seed": int(seed), "auprc": m["auprc_motif_heldout_vs_valid_neg"]})
df = pd.DataFrame(rows)
print("\n=== Wave 4C motif AUPRC ===")
print(df.groupby("variant")["auprc"].agg(["mean", "std", "count"]).round(4).to_string())

# Oracle ceiling from Wave 4B (observed-gate G2/G3)
w3 = Path("outputs/wave3b_motif_2026-07-31")
if w3.exists():
    orb = []
    for p in w3.glob("motif_metrics_*.json"):
        name = p.stem.replace("motif_metrics_", "")
        *vp, seed = name.rsplit("_", 1)
        v = "_".join(vp)
        if v in {"hcr3_full", "hcr3_without_a111", "none", "binary_compact"}:
            m = json.loads(p.read_text())
            orb.append({"variant": f"oracle4b_{v}", "seed": int(seed), "auprc": m["auprc_motif_heldout_vs_valid_neg"]})
    odf = pd.DataFrame(orb)
    print("\n=== Oracle Wave 4B (observed-gate) ===")
    print(odf.groupby("variant")["auprc"].agg(["mean", "std"]).round(4).to_string())

summary = df.groupby("variant")["auprc"].agg(["mean", "std", "count"]).reset_index()
summary.to_csv(d / f"{d.name.split('_')[-1] if False else ''}".strip() or (d.name + "_summary_tmp"), index=False)
# cleaner
day = d.name.replace("wave4c_latent_", "")
summary.to_csv(d / f"{day}_wave4c_LATENT_summary.csv", index=False)
(d / "PROTOCOL.txt").write_text(
    "Wave 4C — Latent-gate HCR recovery\n"
    "Hide A→G from G_train MP; keep B→G.\n"
    "Features for candidate (A,G) use HCR3(A,B,Y) or L1 pairwise A-B|A-Y|B-Y.\n"
    "Gate column G is NEVER read from the patient matrix.\n"
    "Oracle ceiling: Wave 4B G2/G3 motif_metrics (HCR3 with gate).\n"
    "Wave 5 / WNERW is PARKED until Wave 4 is frozen.\n"
)
print(f"\nWrote summary → {d}")
PY

echo "WAVE4C LATENT finished $(date) → ${OUT_DIR}" | tee -a "$LOG"
