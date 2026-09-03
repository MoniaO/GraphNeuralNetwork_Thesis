"""Sealed external Last-FM* benchmark bar chart (NDCG@20 / Recall@20).

Values from LASTFM_EXTERNAL_BENCHMARK_20260824/sealed_test_results/raw/
block_b_*.json — exact frozen artefacts only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures"

# Motif palette (Fig. 6.3): green baselines, orange highlight for proposed
GREEN = "#15803d"
ORANGE = "#c2410c"
EDGE = "#2d3748"

MODELS = [
    "TopPop",
    "UserKNN",
    "ItemKNN",
    r"P3$\alpha$",
    r"RP3$\beta$",
    "Proposed",
]
# Classical baselines: single sealed run. Proposed: mean over seeds 101/202/303.
NDCG = np.array([0.0091, 0.1622, 0.2051, 0.2035, 0.2318, 0.1718])
RECALL = np.array([0.0153, 0.1819, 0.2369, 0.2385, 0.2588, 0.2139])
NDCG_ERR = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0040])
RECALL_ERR = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0024])


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


def _colors(n: int) -> list[str]:
    cols = [GREEN] * (n - 1) + [ORANGE]
    return cols


def main() -> None:
    _style()
    x = np.arange(len(MODELS))
    colors = _colors(len(MODELS))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=False)

    for ax, vals, errs, title, ylim in [
        (axes[0], NDCG, NDCG_ERR, "(a) NDCG@20", (0, 0.30)),
        (axes[1], RECALL, RECALL_ERR, "(b) Recall@20", (0, 0.32)),
    ]:
        ax.bar(
            x,
            vals,
            width=0.68,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            yerr=np.where(errs > 0, errs, np.nan),
            capsize=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": EDGE},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=20, ha="right")
        ax.set_ylabel(title.split()[-1] if False else title[4:])
        ax.set_title(title, loc="left")
        ax.set_ylim(*ylim)
        for i, v in enumerate(vals):
            ax.text(i, v + max(errs[i], 0.004) + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    # Fix y-labels to metric names
    axes[0].set_ylabel("NDCG@20")
    axes[1].set_ylabel("Recall@20")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_lastfm_sealed_benchmark.pdf")
    fig.savefig(OUT / "fig_lastfm_sealed_benchmark.png")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_lastfm_sealed_benchmark.pdf'}")


if __name__ == "__main__":
    main()
