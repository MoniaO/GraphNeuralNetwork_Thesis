#!/usr/bin/env python3
"""E1 — Frozen feature ablation for Task A.

Train on empirical once per (model, seed), then evaluate the same checkpoint on:
  empirical / topology_only / empirical_shuffled

Primary ranking metric: validation AUPRC.
Also reports mean |Δlogit| and Spearman vs empirical logits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.taskA_eval_common import (  # noqa: E402
    MODEL_SPECS,
    SEEDS,
    FEATURE_ABLATION_SEED,
    collect_logits,
    compare_logits,
    evaluate_probabilities,
    load_trained_empirical_model_and_test_data,
    select_threshold_on_validation,
)
from experiments.taskA_interventions import (  # noqa: E402
    feature_fingerprint,
    graph_fingerprint,
    set_all_features_to_ones,
    shuffle_all_node_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "taskA_frozen_feature_ablation.csv",
    )
    args = parser.parse_args()

    rows: list[dict] = []

    for model_key, hydra_model, num_layers in MODEL_SPECS:
        for seed in args.seeds:
            print("=" * 72)
            print(f"E1 {model_key} L{num_layers} seed={seed}")

            model, valid_data, test_data, _train_data, device = (
                load_trained_empirical_model_and_test_data(
                    model_name=model_key,
                    num_layers=num_layers,
                    seed=seed,
                    epochs=args.epochs,
                    force_retrain=args.force_retrain,
                )
            )

            threshold = select_threshold_on_validation(model, valid_data, device)
            g_fp = graph_fingerprint(valid_data)
            c_fp = getattr(valid_data, "candidate_fingerprint", None)

            for split_name, base_data in (("valid", valid_data), ("test", test_data)):
                profiles = {
                    "empirical": base_data,
                    "topology_only": set_all_features_to_ones(base_data),
                    "empirical_shuffled": shuffle_all_node_features(
                        base_data,
                        seed=FEATURE_ABLATION_SEED,
                    ),
                }

                profile_logits: dict[str, object] = {}

                for profile_name, profile_data in profiles.items():
                    labels, logits, probabilities = collect_logits(
                        model, profile_data, device
                    )
                    metrics = evaluate_probabilities(
                        labels, probabilities, threshold=threshold
                    )
                    profile_logits[profile_name] = logits

                    row = {
                        "experiment_name": "E1_frozen_feature_ablation",
                        "intervention": profile_name,
                        "model": model_key,
                        "num_layers": num_layers,
                        "training_seed": seed,
                        "scenario": "clean",
                        "split": split_name,
                        "profile": profile_name,
                        "classification_threshold": threshold,
                        "candidate_fingerprint": c_fp,
                        "feature_fingerprint": feature_fingerprint(profile_data),
                        "graph_fingerprint": g_fp,
                        f"{split_name}_auprc": metrics["auprc"],
                        f"{split_name}_auroc": metrics["auroc"],
                        f"{split_name}_brier": metrics["brier"],
                        f"{split_name}_precision": metrics["precision"],
                        f"{split_name}_recall": metrics["recall"],
                        f"{split_name}_f1": metrics["f1"],
                        f"{split_name}_f2": metrics["f2"],
                        "auprc": metrics["auprc"],
                        "auroc": metrics["auroc"],
                        "brier": metrics["brier"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "f2": metrics["f2"],
                    }
                    rows.append(row)

                empirical_logits = profile_logits["empirical"]
                for profile_name in ("topology_only", "empirical_shuffled"):
                    comparison = compare_logits(
                        empirical_logits,
                        profile_logits[profile_name],
                    )
                    for row in rows:
                        if (
                            row["model"] == model_key
                            and row["training_seed"] == seed
                            and row["split"] == split_name
                            and row["profile"] == profile_name
                        ):
                            row.update(comparison)

    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    # Compact report table on validation
    valid = output[output["split"] == "valid"].copy()
    emp = valid[valid["profile"] == "empirical"][
        ["model", "training_seed", "auprc", "brier"]
    ].rename(columns={"auprc": "auprc_emp", "brier": "brier_emp"})
    for pert in ("topology_only", "empirical_shuffled"):
        sub = valid[valid["profile"] == pert][
            ["model", "training_seed", "auprc", "brier", "mean_abs_logit_change", "spearman_logits"]
        ].rename(
            columns={
                "auprc": f"auprc_{pert}",
                "brier": f"brier_{pert}",
                "mean_abs_logit_change": f"dlogit_{pert}",
                "spearman_logits": f"spearman_{pert}",
            }
        )
        emp = emp.merge(sub, on=["model", "training_seed"], how="left")
        emp[f"delta_auprc_{pert}"] = emp["auprc_emp"] - emp[f"auprc_{pert}"]
        emp[f"delta_brier_{pert}"] = emp[f"brier_{pert}"] - emp["brier_emp"]

    summary_path = args.out.with_name(f"{args.out.stem}_summary.csv")
    emp.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print(emp.groupby("model")[
        [c for c in emp.columns if c.startswith("delta_") or c.startswith("dlogit_") or c.startswith("spearman_")]
    ].mean(numeric_only=True))


if __name__ == "__main__":
    main()
