###
import argparse
from pathlib import Path
import pandas as pd

SCENARIO_FILES = {
    "clean": "synthetic_pharmacotherapy_v2_samples_clean.csv",
    "hidden_confounder": "synthetic_pharmacotherapy_v2_samples_hidden_confounder.csv",
    "selection_bias": "synthetic_pharmacotherapy_v2_samples_selection_bias.csv",
    "no_overlap": "synthetic_pharmacotherapy_v2_samples_no_overlap.csv",
    "noisy_documentation": "synthetic_pharmacotherapy_v2_samples_noisy_documentation.csv",
    "multihospital": "synthetic_pharmacotherapy_v2_samples_multihospital.csv",
}

DEFAULT_KEEP_SCENARIO = False


def read_scenario_frame(root: Path, scenario: str, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing scenario file: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "benchmark_scenario" not in df.columns:
        df["benchmark_scenario"] = scenario
    return df


def normalize_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "benchmark_split" not in df.columns:
        raise ValueError(
            "No 'benchmark_split' column found. Build the benchmark first with build_v2_1_nn_benchmark.py "
            "or add benchmark_split before combining."
        )

    out = df.copy()
    mapping = {
        "val": "valid",
        "validation": "valid",
        "valid": "valid",
        "train": "train",
        "test": "test",
        "internal_test": "test",
        "external_test": "test",
    }
    out["benchmark_split"] = out["benchmark_split"].astype(str).str.strip().str.lower().map(mapping)
    unknown = sorted(out.loc[out["benchmark_split"].isna(), "benchmark_split"].dropna().unique().tolist())
    if out["benchmark_split"].isna().any():
        bad_rows = df.loc[out["benchmark_split"].isna(), [c for c in ["benchmark_scenario", "benchmark_split"] if c in df.columns]]
        raise ValueError(f"Unknown benchmark_split values detected. Example rows:\n{bad_rows.head(10).to_string(index=False)}")
    return out


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [c for c in ["patient_id", "hospital_id", "benchmark_scenario", "benchmark_split", "replicate_id", "generation_seed", "paired_patient_key"] if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def build_outputs(source_dir: Path, out_dir: Path, keep_scenario: bool) -> None:
    frames = []
    for scenario, filename in SCENARIO_FILES.items():
        frames.append(read_scenario_frame(source_dir, scenario, filename))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = normalize_split_labels(combined)

    if not keep_scenario and "benchmark_scenario" in combined.columns:
        combined_no_scenario = combined.drop(columns=["benchmark_scenario"])
    else:
        combined_no_scenario = combined.copy()

    combined_no_scenario = reorder_columns(combined_no_scenario)

    train_df = combined_no_scenario.loc[combined_no_scenario["benchmark_split"] == "train"].copy()
    valid_df = combined_no_scenario.loc[combined_no_scenario["benchmark_split"] == "valid"].copy()
    test_df = combined_no_scenario.loc[combined_no_scenario["benchmark_split"] == "test"].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_no_scenario.to_csv(out_dir / "benchmark_combined_all.csv", index=False)
    train_df.to_csv(out_dir / "benchmark_combined_train.csv", index=False)
    valid_df.to_csv(out_dir / "benchmark_combined_valid.csv", index=False)
    test_df.to_csv(out_dir / "benchmark_combined_test.csv", index=False)

    summary_rows = []
    for name, frame in [("all", combined_no_scenario), ("train", train_df), ("valid", valid_df), ("test", test_df)]:
        row = {"split": name, "rows": len(frame), "columns": frame.shape[1]}
        if "hospital_id" in frame.columns:
            row["unique_hospitals"] = frame["hospital_id"].nunique(dropna=True)
        if "replicate_id" in frame.columns:
            row["unique_replicates"] = frame["replicate_id"].nunique(dropna=True)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "benchmark_combined_manifest.csv", index=False)
