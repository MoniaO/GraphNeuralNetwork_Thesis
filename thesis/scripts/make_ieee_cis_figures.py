"""IEEE-CIS-G (H2GB) result table helper and thesis figures.

Own runs: five seeds 42, 202, 568, 20256, 202567 (l1 logs).
H2GB baselines: Lin et al., KDD 2025, Table 3, IEEE-CIS-G F1 (5 seeds), converted to [0,1].
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = Path("/Users/monika/IMDB/logs/l1")
CSVS = [
    LOG_DIR / "ieee_sweep_summary_seeds20256_202567_42.csv",
    LOG_DIR / "ieee_sweep_summary_seed202.csv",
    LOG_DIR / "ieee_sweep_summary_seed568.csv",
]
OUT = Path(__file__).resolve().parents[1] / "figures"

# Last-FM sealed palette
GREEN = "#15803d"
ORANGE = "#c2410c"
NAVY = "#1e3a5f"
EDGE = "#2d3748"

# H2GB Table 3, IEEE-CIS-G column, F1 as percent → fraction
H2GB_F1 = {
    "MLP": (0.0426, 0.0852),
    "GCN": (0.2879, 0.0107),
    "GraphSAGE": (0.3149, 0.0123),
    "GAT": (0.2851, 0.0045),
    "HGT": (0.3089, 0.0080),
    "R-GCN": (0.3144, 0.0096),
    "SHGN": (0.3166, 0.0086),
    r"H$^2$G-former": (0.3155, 0.0092),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def load_runs() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(path) for path in CSVS], ignore_index=True)
    df["variant"] = np.where(
        df["HCR"] == "tak",
        "HCR 40d",
        "---",
    )
    df["res"] = np.where(df["residual"] == "tak", "yes", "no")
    return df


def _agg(g: pd.DataFrame, col: str) -> tuple[float, float, int]:
    x = pd.to_numeric(g[col], errors="coerce").dropna()
    n = int(x.shape[0])
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(x.iloc[0]), float("nan"), 1
    return float(x.mean()), float(x.std(ddof=1)), n


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for L in (2, 3, 6):
        for res, variant in (("no", "---"), ("yes", "---"), ("yes", "HCR 40d")):
            g = df[(df["L"] == L) & (df["res"] == res) & (df["variant"] == variant)]
            row = {"L": L, "res": res, "variant": variant, "n": len(g)}
            for col in (
                "val_f1_bin",
                "val_auc",
                "val_auprc",
                "test_f1_bin",
                "test_auc",
                "test_auprc",
                "test_f1_bin_tuned",
            ):
                mean, std, n = _agg(g, col)
                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
                row["n"] = n
            rows.append(row)
    return pd.DataFrame(rows)


def msd(mean: float, std: float, nd: int = 3) -> str:
    if not np.isfinite(mean):
        return "---"
    if not np.isfinite(std):
        return f"{mean:.{nd}f}"
    return f"${mean:.{nd}f}{{\\scriptstyle\\,\\pm\\,{std:.{nd}f}}}$"


def latex_table(stats: pd.DataFrame) -> str:
    lines = []
    for L, block in stats.groupby("L", sort=True):
        if L != 2:
            lines.append(r"    \midrule")
        first = True
        for _, r in block.iterrows():
            if not first and r["res"] == "yes" and r["variant"] == "---":
                lines.append(r"    \addlinespace[2pt]")
            first = False
            cells = [
                msd(r["val_f1_bin_mean"], r["val_f1_bin_std"]),
                msd(r["val_auc_mean"], r["val_auc_std"]),
                msd(r["val_auprc_mean"], r["val_auprc_std"]),
                msd(r["test_f1_bin_mean"], r["test_f1_bin_std"]),
                msd(r["test_auc_mean"], r["test_auc_std"]),
                msd(r["test_auprc_mean"], r["test_auprc_std"]),
            ]
            lines.append(
                f"    {int(r['L'])} & {r['res']} & {r['variant']:8} & "
                + " & ".join(cells)
                + r" \\"
            )
    return "\n".join(lines)


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def plot_ablation(stats: pd.DataFrame) -> None:
    _style()
    layers = [2, 3, 6]
    series = [
        ("no", "---", "Graph-only, no residual", "#94a3b8"),
        ("yes", "---", "Graph-only, residual", NAVY),
        ("yes", "HCR 40d", "HCR 40d + residual", ORANGE),
    ]
    x = np.arange(len(layers))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for ax, metric, ylabel, title, ylim in [
        (axes[0], "test_f1_bin", "Test F1 (binary)", "(a) Test F1", (0, 0.72)),
        (axes[1], "test_auc", "Test AUROC", "(b) Test AUROC", (0.50, 0.95)),
    ]:
        for i, (res, variant, label, color) in enumerate(series):
            means, stds = [], []
            for L in layers:
                r = stats[
                    (stats["L"] == L)
                    & (stats["res"] == res)
                    & (stats["variant"] == variant)
                ].iloc[0]
                means.append(r[f"{metric}_mean"])
                stds.append(r[f"{metric}_std"])
            ax.bar(
                x + (i - 1) * width,
                means,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                yerr=stds,
                capsize=3,
                error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": EDGE},
                label=label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"$L={L}$" for L in layers])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_ylim(*ylim)

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_ieee_cis_ablation")


def plot_h2gb_comparison(stats: pd.DataFrame) -> None:
    _style()
    # Proposed: HCR + residual at L=3 (H2GB HGT protocol depth)
    prop = stats[
        (stats["L"] == 3) & (stats["res"] == "yes") & (stats["variant"] == "HCR 40d")
    ].iloc[0]
    names = list(H2GB_F1.keys()) + ["Proposed\n(HCR, $L=3$)"]
    means = [H2GB_F1[k][0] for k in H2GB_F1] + [prop["test_f1_bin_mean"]]
    stds = [H2GB_F1[k][1] for k in H2GB_F1] + [prop["test_f1_bin_std"]]
    colors = [GREEN] * len(H2GB_F1) + [ORANGE]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.bar(
        x,
        means,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        yerr=stds,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": EDGE},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Test F1 (binary)")
    ax.set_ylim(0, 0.72)
    for i, (v, e) in enumerate(zip(means, stds)):
        ax.text(i, v + e + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig_ieee_cis_h2gb_comparison")


def main() -> None:
    df = load_runs()
    print("seeds:", sorted(df["seed"].unique().tolist()), "n_rows=", len(df))
    stats = summarize(df)
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n% --- latex rows ---")
    print(latex_table(stats))
    plot_ablation(stats)
    plot_h2gb_comparison(stats)


if __name__ == "__main__":
    main()
