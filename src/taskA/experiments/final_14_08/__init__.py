"""FINAL 14.08 — frozen stack, 6 scenarios × 5 seeds × MLP/KAN.

What it does
------------
Freeze constants from Stage A/C. The runner launches 60 fresh jobs into
`outputs/taskA_FINAL_14.08.2026/` (does not overwrite the 11.08 campaign).

What you may change
-------------------
Nothing if you reproduce the 14.08 table. A new experiment = a new package
(e.g. `experiments/final_XX_YY/`), not an edit of `FROZEN` below.

Script: `scripts/taskA/08_run_final.py`.
"""

BLOCK_DATE = "14.08.2026"
BLOCK_ID = "taskA_FINAL_14.08.2026"
MODEL_NAME = "FINAL_14.08.2026"
CANDIDATE_SEED = 20260722
FINAL_SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)
FROZEN = {
    "backbone": "hgt",
    "hidden_dim": 32,
    "n_layers": 2,
    "dropout": 0.25,
    "lr": 1e-3,
    "heads": 4,
    "weight_decay": 5e-4,
    "epochs": 200,
    "patience": 40,
    "grad_clip": 1.0,
    "stat_variant": "S10_HCR_FULL40",
    "decoder": "fusion88_stat",
}
