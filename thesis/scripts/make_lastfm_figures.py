"""Generate Last-FM* ablation bar charts for the thesis.

Single navy accent (#1e3a5f), matching Fig. 6.3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures"

# Official means (3 seeds): sampled checkpoint metric + full-catalog evaluation
VARIANTS = [
    "HGT only",
    "HGT+A5",
    "HGT+A5+H3",
    "Full model\n(+LEG)",
]
SAMPLED = np.array([0.4751, 0.7908, 0.8729, 0.8733])
FULLRANK = np.array([0.0093, 0.0640, 0.2321, 0.2764])
# per-seed fullrank for error bars (std)
FULLRANK_STD = np.array([0.0007, 0.0029, 0.0080, 0.0041])
SAMPLED_STD = np.array([0.0005, 0.0003, 0.0001, 0.0002])  # approx from seed spread

# Single accent colour (same navy as Fig. 6.3)
BAR_COLOR = "#1e3a5f"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def main() -> None:
    _style()
    x = np.arange(len(VARIANTS))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)

    # Panel (a): sampled
    ax = axes[0]
    ax.bar(
        x,
        SAMPLED,
        width=0.62,
        color=BAR_COLOR,
        edgecolor="white",
        linewidth=0.6,
        yerr=SAMPLED_STD,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("NDCG@20")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Sampled validation (checkpoint metric)")
    for i, v in enumerate(SAMPLED):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Panel (b): full-catalog
    ax = axes[1]
    ax.bar(
        x,
        FULLRANK,
        width=0.62,
        color=BAR_COLOR,
        edgecolor="white",
        linewidth=0.6,
        yerr=FULLRANK_STD,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("NDCG@20")
    ax.set_ylim(0, 0.35)
    ax.set_title("(b) Full-catalog development (evaluation only)")
    for i, v in enumerate(FULLRANK):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Last-FM*: progressive explicit context — sampled vs full-catalog NDCG@20",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_lastfm_ablation_bars")
    print(f"wrote {OUT / 'fig_lastfm_ablation_bars.pdf'}")


if __name__ == "__main__":
    main()
