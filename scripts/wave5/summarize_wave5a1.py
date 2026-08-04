#!/usr/bin/env python3
"""Wave 5A.1 — Panel A (all frozen queries) + Panel B (common reachable set)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

METRICS = [
    "true_path_mass",
    "reciprocal_rank",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "path_entropy",
    "hub_mass",
]


def load_query_metrics(eval_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(eval_dir.glob("query_metrics_*.csv")):
        frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit(f"No query_metrics_*.csv in {eval_dir}")
    return pd.concat(frames, ignore_index=True)


def panel_a(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub in df.groupby("variant"):
        row = {
            "variant": variant,
            "n_query_seed": int(len(sub)),
            "path_coverage": float(sub["has_any_path"].mean()),
            "true_path_coverage": float(sub["has_true_path"].mean()),
            "unconditional_tpm_mean": float(sub["true_path_mass"].mean()),
            "unconditional_tpm_std": float(sub["true_path_mass"].std(ddof=0)),
            "unconditional_mrr": float(sub["reciprocal_rank"].mean()),
            "unconditional_hit_at_1": float(sub["hit_at_1"].mean()),
            "unconditional_hit_at_5": float(sub["hit_at_5"].mean()),
            "unconditional_hit_at_10": float(sub["hit_at_10"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("unconditional_tpm_mean", ascending=False)


def common_mask(
    df: pd.DataFrame,
    *,
    variants: list[str] | None = None,
) -> pd.Series:
    """Per-seed: query is common if every selected variant has has_any_path."""
    sub = df if variants is None else df[df["variant"].isin(variants)]
    flags = (
        sub.groupby(["seed", "query_id", "variant"])["has_any_path"]
        .any()
        .unstack("variant")
    )
    if variants is not None:
        flags = flags.reindex(columns=list(variants))
    common = flags.fillna(False).all(axis=1)
    key = df.set_index(["seed", "query_id"]).index
    return key.map(common.astype(bool))


def panel_b(
    df: pd.DataFrame,
    *,
    variants: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = common_mask(df, variants=variants)
    common = df.loc[mask].copy()
    if variants is not None:
        common = common[common["variant"].isin(variants)]
    rows = []
    for variant, sub in common.groupby("variant"):
        row = {
            "variant": variant,
            "n_query_seed": int(len(sub)),
            "paired_tpm_mean": float(sub["true_path_mass"].mean()),
            "paired_tpm_std": float(sub["true_path_mass"].std(ddof=0)),
            "paired_mrr": float(sub["reciprocal_rank"].mean()),
            "paired_hit_at_1": float(sub["hit_at_1"].mean()),
            "paired_hit_at_5": float(sub["hit_at_5"].mean()),
            "paired_hit_at_10": float(sub["hit_at_10"].mean()),
            "paired_entropy_mean": float(sub["path_entropy"].mean()),
            "paired_hub_mass_mean": float(sub["hub_mass"].mean()),
        }
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("paired_tpm_mean", ascending=False)

    # Paired deltas vs reference (prefer hcr2_finite, else first non-shuffle).
    used = list(summary["variant"])
    ref = "hcr2_finite" if "hcr2_finite" in used else (used[0] if used else "")
    wide = common.pivot_table(
        index=["seed", "query_id"],
        columns="variant",
        values="true_path_mass",
        aggfunc="first",
    )
    delta_rows = []
    for v in used:
        if v == ref or v not in wide.columns or ref not in wide.columns:
            continue
        d = (wide[ref] - wide[v]).dropna()
        delta_rows.append(
            {
                "reference": ref,
                "variant": v,
                "n_paired": int(len(d)),
                "delta_tpm_mean": float(d.mean()),
                "delta_tpm_std": float(d.std(ddof=0)),
                "frac_ref_better": float((d > 0).mean()) if len(d) else float("nan"),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    return summary, deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=ROOT / "outputs/wave5/eval_5a1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/wave5",
    )
    args = parser.parse_args()

    df = load_query_metrics(args.eval_dir)
    a = panel_a(df)
    # Exclude train-only uniform from the paired common set — different graph family.
    predicted_variants = sorted(
        v for v in df["variant"].unique() if v != "uniform"
    )
    b, deltas = panel_b(df, variants=predicted_variants)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    a_path = args.out_dir / "WAVE5A1_panel_A_all_queries.csv"
    b_path = args.out_dir / "WAVE5A1_panel_B_common_queries.csv"
    d_path = args.out_dir / "WAVE5A1_panel_B_paired_deltas.csv"
    a.to_csv(a_path, index=False)
    b.to_csv(b_path, index=False)
    deltas.to_csv(d_path, index=False)

    print("=== PANEL A — all frozen queries (unconditional) ===")
    print(a.to_string(index=False))
    print(
        "\n=== PANEL B — common reachable set among predicted-graph variants ==="
    )
    print(f"(variants: {predicted_variants})")
    print(b.to_string(index=False))
    if len(deltas):
        print("\n=== PANEL B — paired TPM deltas (ref − variant) ===")
        print(deltas.to_string(index=False))
    print(f"\nWrote {a_path}")
    print(f"Wrote {b_path}")
    print(f"Wrote {d_path}")


if __name__ == "__main__":
    main()
