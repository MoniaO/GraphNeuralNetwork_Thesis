"""Patient matrix + split. Fit node features and S10 on train rows only.

What you may change: paths in `cfg.data.dataset`. Do not mix valid/test into the fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def resolve_dataset_paths(cfg: Any) -> dict[str, Path]:
    data_dir = Path(str(cfg.data.dataset.root_dir)).expanduser().resolve()
    scenario = str(cfg.data.dataset.scenario)
    samples_map = dict(cfg.data.dataset.samples)
    if scenario not in samples_map:
        raise KeyError(f"Unknown scenario {scenario!r} in cfg.data.dataset.samples")
    samples_path = data_dir / str(samples_map[scenario])
    split_name = str(cfg.data.dataset.patient_split_file)
    split_path = data_dir.parent / "splits" / split_name
    nodes_path = data_dir / str(cfg.data.dataset.nodes_file)
    return {
        "data_dir": data_dir,
        "samples_path": samples_path,
        "split_path": split_path,
        "nodes_path": nodes_path,
    }


def load_patient_matrix_with_split(cfg: Any) -> pd.DataFrame:
    """Load scenario samples joined with patient split labels."""
    paths = resolve_dataset_paths(cfg)
    patients = pd.read_csv(paths["samples_path"])
    splits = pd.read_csv(paths["split_path"])

    if "patient_id" not in patients.columns:
        raise KeyError("Patient samples must contain column 'patient_id'.")
    if not {"patient_id", "split"}.issubset(splits.columns):
        raise KeyError("Patient split file must contain patient_id and split.")

    merged = patients.merge(splits[["patient_id", "split"]], on="patient_id", how="inner")
    if merged.empty:
        raise RuntimeError("No patients remain after joining samples with splits.")
    return merged


def train_patient_df(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Return only training patients (canonical HCR fit partition)."""
    if "split" not in patient_df.columns:
        raise KeyError("patient_df must contain a 'split' column.")
    # Accept both 'train' and legacy aliases.
    mask = patient_df["split"].astype(str).str.lower().isin({"train", "training"})
    out = patient_df.loc[mask].copy()
    if out.empty:
        raise RuntimeError("Training patient partition is empty.")
    return out
