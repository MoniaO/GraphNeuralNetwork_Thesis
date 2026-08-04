#!/usr/bin/env python3
"""E2 — Node-type permutation importance (frozen empirical checkpoint)."""

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
    shuffle_selected_node_types,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "taskA_node_type_permutation.csv",
    )
    args = parser.parse_args()

    rows: list[dict] = []

    for model_key, _hydra_model, num_layers in MODEL_SPECS:
        for seed in args.seeds:
            print("=" * 72)
            print(f"E2 {model_key} L{num_layers} seed={seed}")

            model, valid_data, test_data, _train, device = (
                load_trained_empirical_model_and_test_data(
                    model_name=model_key,
                    num_layers=num_layers,
                    seed=seed,
                    epochs=args.epochs,
                    force_retrain=args.force_retrain,
                )
            )
            threshold = select_threshold_on_validation(model, valid_data, device)

            for split_name, data in (("valid", valid_data), ("test", test_data)):
                labels, reference_logits, reference_probabilities = collect_logits(
                    model, data, device
                )
                reference_metrics = evaluate_probabilities(
                    labels, reference_probabilities, threshold=threshold
                )

                for node_type in sorted(data.node_types):
                    perturbed = shuffle_selected_node_types(
                        data,
                        node_types=[node_type],
                        seed=FEATURE_ABLATION_SEED,
                    )
                    _plabels, perturbed_logits, perturbed_probabilities = collect_logits(
                        model, perturbed, device
                    )
                    perturbed_metrics = evaluate_probabilities(
                        labels, perturbed_probabilities, threshold=threshold
                    )
                    comparison = compare_logits(reference_logits, perturbed_logits)

                    rows.append(
                        {
                            "experiment_name": "E2_node_type_permutation",
                            "intervention": f"shuffle_{node_type}",
                            "model": model_key,
                            "num_layers": num_layers,
                            "training_seed": seed,
                            "scenario": "clean",
                            "split": split_name,
                            "node_type": node_type,
                            "classification_threshold": threshold,
                            "candidate_fingerprint": getattr(
                                data, "candidate_fingerprint", None
                            ),
                            "feature_fingerprint": feature_fingerprint(perturbed),
                            "graph_fingerprint": graph_fingerprint(data),
                            "reference_auprc": reference_metrics["auprc"],
                            "perturbed_auprc": perturbed_metrics["auprc"],
                            "delta_auprc": (
                                reference_metrics["auprc"] - perturbed_metrics["auprc"]
                            ),
                            "reference_brier": reference_metrics["brier"],
                            "perturbed_brier": perturbed_metrics["brier"],
                            **comparison,
                        }
                    )

    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    valid = output[output["split"] == "valid"]
    pivot = (
        valid.groupby(["model", "node_type"])["delta_auprc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_path = args.out.with_name("taskA_node_type_permutation_summary.csv")
    pivot.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
