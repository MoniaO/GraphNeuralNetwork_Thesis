#!/usr/bin/env python3
"""Audit Task A feature ablations before FEATURE_ABLATION training runs.

Checks:
  - candidate fingerprint identical across profiles (G-TRAIN / CAND frozen);
  - empirical fingerprint != topology_only / empirical_shuffled;
  - topology_only uses all-ones features;
  - empirical_shuffled preserves marginal mean/std but changes fingerprint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.PreprocessingTaskA.feature_ablation import (  # noqa: E402
    feature_summary,
    mean_absolute_feature_difference,
)
from data.PreprocessingTaskA.load_hetero_recon_data import (  # noqa: E402
    load_recon_heterodata,
)


PROFILES = ("empirical", "topology_only", "empirical_shuffled")


def load_profile(profile: str, model: str = "TaskA_hetero_sage", num_layers: int = 2):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model}",
                "data.dataset.scenario=clean",
                f"data.feature_ablation_profile={profile}",
                "data.feature_ablation_seed=20260722",
                "data.candidate_seed=20260722",
                "training.seed=20260721",
                f"model.num_layers={num_layers}",
                "wandb.enabled=false",
            ],
        )
    train_data, valid_data, test_data, _ = load_recon_heterodata(cfg)
    return cfg, train_data, valid_data, test_data


def main() -> None:
    loaded = {}
    for profile in PROFILES:
        print("=" * 72)
        print(f"Loading profile={profile}")
        _, train_data, valid_data, test_data = load_profile(profile)
        loaded[profile] = {
            "train": train_data,
            "valid": valid_data,
            "test": test_data,
        }
        print(
            f"  candidate_fingerprint={train_data.candidate_fingerprint}"
            f"\n  feature_fingerprint={train_data.feature_fingerprint}"
        )

    empirical = loaded["empirical"]["train"]
    topology = loaded["topology_only"]["train"]
    shuffled = loaded["empirical_shuffled"]["train"]

    cand_fps = {
        profile: loaded[profile]["train"].candidate_fingerprint for profile in PROFILES
    }
    feat_fps = {
        profile: loaded[profile]["train"].feature_fingerprint for profile in PROFILES
    }

    errors: list[str] = []

    if len(set(cand_fps.values())) != 1:
        errors.append(f"Candidate fingerprints differ across profiles: {cand_fps}")

    if feat_fps["empirical"] == feat_fps["topology_only"]:
        errors.append("empirical and topology_only share the same feature fingerprint")

    if feat_fps["empirical"] == feat_fps["empirical_shuffled"]:
        errors.append("empirical and empirical_shuffled share the same feature fingerprint")

    # topology_only must be all ones
    for node_type in topology.node_types:
        x = topology[node_type].x
        if not torch.allclose(x, torch.ones_like(x)):
            errors.append(f"topology_only features are not all-ones for {node_type}")

    emp_vs_top = mean_absolute_feature_difference(empirical, topology)
    emp_vs_shuf = mean_absolute_feature_difference(empirical, shuffled)
    if emp_vs_top <= 0:
        errors.append(f"empirical vs topology_only MAD should be > 0, got {emp_vs_top}")
    if emp_vs_shuf <= 0:
        errors.append(f"empirical vs shuffled MAD should be > 0, got {emp_vs_shuf}")

    # Marginal stats roughly preserved under row shuffle (per type mean/std)
    emp_summary = feature_summary(empirical)
    shuf_summary = feature_summary(shuffled)
    for node_type in emp_summary:
        mean_delta = abs(emp_summary[node_type]["mean"] - shuf_summary[node_type]["mean"])
        std_delta = abs(emp_summary[node_type]["std"] - shuf_summary[node_type]["std"])
        if mean_delta > 1e-5 or std_delta > 1e-5:
            errors.append(
                f"shuffled changed marginal stats for {node_type}: "
                f"meanΔ={mean_delta}, stdΔ={std_delta}"
            )

    # Same features across train/valid/test within a profile (transductive)
    for profile in PROFILES:
        t = loaded[profile]["train"]
        v = loaded[profile]["valid"]
        te = loaded[profile]["test"]
        if t.feature_fingerprint != v.feature_fingerprint or t.feature_fingerprint != te.feature_fingerprint:
            errors.append(f"{profile}: train/valid/test feature fingerprints differ")

    print("\n" + "=" * 72)
    print("candidate fingerprints:", cand_fps)
    print("feature fingerprints:  ", feat_fps)
    print(f"MAD empirical vs topology_only: {emp_vs_top:.6f}")
    print(f"MAD empirical vs shuffled:      {emp_vs_shuf:.6f}")

    if errors:
        print("\nAUDIT FAILED")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print(
        "\nAUDIT PASSED\n"
        "The graph and candidate pairs are frozen, while only node features "
        "change across ablation profiles."
    )


if __name__ == "__main__":
    main()
