"""FINAL 14.08 — zamrożony stack, 6 scenariuszy × 5 seedów × MLP/KAN.

Co robi
-------
Stałe freeze z Stage A/C. Runner odpala 60 świeżych jobów do
`outputs/taskA_FINAL_14.08.2026/` (nie nadpisuje kampanii 11.08).

Co wolno zmieniać
-----------------
Nic, jeśli odtwarzasz tabelę 14.08. Nowy eksperyment = nowy pakiet
(np. `experiments/final_XX_YY/`), nie edycja `FROZEN` poniżej.

Skrypt: `scripts/taskA/08_run_final.py`.
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
