#!/usr/bin/env python3
"""E3 — Per-edge-type metrics under empirical / topology_only / shuffled features.

Uses frozen empirical checkpoints. Threshold is selected on validation once
per (model, seed) and reused for all groups / profiles / test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.taskA_group_metrics import (  # noqa: E402
    relation_metrics_with_shared_negatives,
)
from evaluation.taskA_per_edge_type_summaries import (  # noqa: E402
    write_per_edge_type_summaries,
)
from experiments.taskA_eval_common import (  # noqa: E402
    MODEL_SPECS,
    SEEDS,
    FEATURE_ABLATION_SEED,
    collect_logits,
    load_trained_empirical_model_and_test_data,
    select_threshold_on_validation,
)
from experiments.taskA_interventions import (  # noqa: E402
    set_all_features_to_ones,
    shuffle_all_node_features,
)


def prediction_frame(data, labels, probabilities) -> pd.DataFrame:
    edge_types = getattr(data, "candidate_edge_type", None)
    source_types = getattr(data, "candidate_source_node_type", None)
    target_types = getattr(data, "candidate_target_node_type", None)

    if edge_types is None:
        raise AttributeError(
            "candidate_edge_type missing on HeteroData — update the loader."
        )

    frame = pd.DataFrame(
        {
            "label": labels,
            "probability": probabilities,
            "edge_type": list(edge_types),
            "source_node_type": list(source_types) if source_types is not None else None,
            "target_node_type": list(target_types) if target_types is not None else None,
        }
    )
    # Analysis focuses on audited relation types (positives carry edge_type).
    # Keep negatives too, but mark them.
    frame["is_negative_candidate"] = frame["edge_type"].eq("negative")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "taskA_per_edge_type.csv",
    )
    args = parser.parse_args()

    rows: list[pd.DataFrame] = []

    for model_key, _hydra, num_layers in MODEL_SPECS:
        for seed in args.seeds:
            print("=" * 72)
            print(f"E3 {model_key} L{num_layers} seed={seed}")

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

            for split_name, base_data in (("valid", valid_data), ("test", test_data)):
                profiles = {
                    "empirical": base_data,
                    "topology_only": set_all_features_to_ones(base_data),
                    "empirical_shuffled": shuffle_all_node_features(
                        base_data, seed=FEATURE_ABLATION_SEED
                    ),
                }

                for profile_name, profile_data in profiles.items():
                    labels, _logits, probabilities = collect_logits(
                        model, profile_data, device
                    )
                    frame = prediction_frame(profile_data, labels, probabilities)

                    # Positives of relation r + shared negatives → well-defined AUPRC.
                    grouped = relation_metrics_with_shared_negatives(
                        frame,
                        threshold=threshold,
                        edge_type_column="edge_type",
                    )
                    grouped.insert(0, "experiment_name", "E3_per_edge_type")
                    grouped.insert(1, "model", model_key)
                    grouped.insert(2, "num_layers", num_layers)
                    grouped.insert(3, "training_seed", seed)
                    grouped.insert(4, "scenario", "clean")
                    grouped.insert(5, "split", split_name)
                    grouped.insert(6, "profile", profile_name)
                    grouped.insert(7, "classification_threshold", threshold)
                    grouped.insert(
                        8,
                        "candidate_fingerprint",
                        getattr(base_data, "candidate_fingerprint", None),
                    )
                    rows.append(grouped)

    output = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    # Do NOT filter on per-seed insufficient_support (n_positive >= 10):
    # on this split almost every relation fails that bar. Summaries use
    # exploratory (mean n_pos >= 4) and main (pooled n_pos >= 20) rules.
    paths = write_per_edge_type_summaries(args.out, split="valid")
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
