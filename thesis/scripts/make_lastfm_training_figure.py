"""Last-FM* development training-dynamics figure for the thesis.

Sources
-------
- TRUE FINAL joint training seed 303 ``train_meta.json`` (full per-epoch history).
- Freeze epochs for seeds 101/202/303 from the published seed table
  (sampled-validation NDCG@20 selection on model_train → validation).

Does not use sealed-test scores. Protocol stages are described in the surrounding prose,
not drawn as figure panels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures"
KGAT = Path("/Users/martajasiewicz/Desktop/KGAT_KnowledgeGraphs")
TRAIN_META = (
    KGAT
    / "LASTFM_TRUE_FINAL/JOINT_TRAINING_V1/03_SEED303/train_meta.json"
)

# Freeze epochs under development sampled NDCG@20 (thesis seed table).
FREEZE = {101: 36, 202: 34, 303: 43}

# Same palette as Fig. 6.3 (context diagram)
NAVY = "#1e3a5f"
TEAL = "#0f766e"
ORANGE = "#c2410c"
GREEN = "#15803d"
EDGE = "#2d3748"

COL_LOSS = NAVY
COL_VAL = ORANGE
COL_MARK = GREEN


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
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


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


def _load_seed303():
    meta = json.loads(TRAIN_META.read_text(encoding="utf-8"))
    hist = meta["history"]
    epochs = np.array([int(h["epoch"]) for h in hist], dtype=int)
    loss = np.array([float(h["loss"]) for h in hist], dtype=float)
    val = np.array([float(h["val_NDCG@20"]) for h in hist], dtype=float)
    best = int(meta["best_epoch"])
    return epochs, loss, val, best


def main() -> None:
    _style()
    epochs, loss, val, best = _load_seed303()
    assert best == FREEZE[303], f"seed-303 best_epoch={best} != freeze table {FREEZE[303]}"

    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    ax.plot(epochs, loss, color=COL_LOSS, lw=1.8, label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss", color=COL_LOSS)
    ax.tick_params(axis="y", labelcolor=COL_LOSS)
    ax.set_xlim(0, int(epochs.max()) + 1)
    ax.set_ylim(0.35, 1.2)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(epochs, val, color=COL_VAL, lw=1.8, label="Sampled val NDCG@20")
    ax2.set_ylabel("Sampled validation NDCG@20", color=COL_VAL)
    ax2.tick_params(axis="y", labelcolor=COL_VAL)
    ax2.set_ylim(0.65, 0.90)

    ax.axvline(best, color=COL_MARK, ls="--", lw=1.25, zorder=3)
    ax2.scatter(
        [best],
        [val[best]],
        color=COL_MARK,
        s=42,
        zorder=4,
        edgecolors="white",
        linewidths=0.7,
    )

    note = "Frozen epochs:  " + "  ·  ".join(
        f"{s}→{e}" for s, e in FREEZE.items()
    )
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=EDGE,
        alpha=0.85,
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=False,
        borderaxespad=0.6,
    )

    fig.tight_layout()
    _save(fig, "fig_lastfm_training_protocol")


if __name__ == "__main__":
    main()
