"""Generate architecture pipeline figures for the thesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures"

COL_GRAPH = "#D6E4F7"
COL_STAT = "#FCE8D5"
COL_FUSE = "#E2F0E2"
COL_EDGE = "#374151"
COL_ARROW = "#4B5563"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc, fs=8.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0,
        edgecolor=COL_EDGE,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=COL_ARROW,
            shrinkA=2,
            shrinkB=2,
        )
    )


def fig_taskA_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(5, 4.65, "Task A — frozen pipeline", ha="center", fontsize=11, fontweight="bold")

    _box(ax, 0.2, 3.5, 1.5, 0.75, r"$G_{\mathrm{train}}$", COL_GRAPH)
    _box(ax, 2.0, 3.35, 1.7, 1.05, "HGT\n$d{=}32,L{=}2,H{=}4$", COL_GRAPH)
    _box(ax, 4.0, 3.35, 1.8, 1.05, r"$q_{AG}$" + "\n128D", COL_GRAPH)
    _box(ax, 6.0, 3.35, 1.5, 1.05, r"$g_{\mathrm{graph}}$" + "\n64D", COL_GRAPH)

    _box(ax, 0.2, 1.55, 1.8, 0.85, r"Motif $AZ,AG,ZG$", COL_STAT)
    _box(ax, 2.2, 1.55, 1.7, 0.85, "HCR-inspired\n40D × 3", COL_STAT)
    _box(ax, 4.1, 1.55, 1.8, 0.85, "Role encoders\n40→8", COL_STAT)
    _box(ax, 6.1, 1.55, 1.4, 0.85, r"$g_{\mathrm{stat}}$" + "\n24D", COL_STAT)

    _box(ax, 7.8, 2.35, 1.8, 1.0, "Fusion88\n88→64→1", COL_FUSE)
    _box(ax, 7.95, 0.55, 1.5, 0.7, r"logit $A{\to}G$", "#FFFFFF")

    for x in (1.7, 3.8, 5.8, 7.5):
        _arrow(ax, x, 3.85, x + 0.18, 3.85)
    for x in (2.0, 3.9, 5.9, 7.5):
        _arrow(ax, x, 1.95, x + 0.18, 1.95)
    _arrow(ax, 6.75, 3.85, 7.75, 3.0)
    _arrow(ax, 6.75, 1.95, 7.75, 2.7)
    _arrow(ax, 8.7, 2.35, 8.7, 1.3)

    ax.text(5, 2.75, "graph branch", ha="center", fontsize=8, color="#1D4ED8")
    ax.text(3.8, 1.15, "statistical branch (HCR-inspired)", ha="center", fontsize=8, color="#C2410C")
    _save(fig, "fig_taskA_pipeline")


def fig_taskA_motif() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.text(5, 4.2, r"Role-aware motif for candidate edge $A \to G$", ha="center", fontsize=11, fontweight="bold")

    _box(ax, 1.0, 2.6, 1.2, 0.7, r"$A$", "#FFFFFF", fs=10)
    _box(ax, 4.4, 2.6, 1.2, 0.7, r"$Z$", "#FFFFFF", fs=10)
    _box(ax, 7.8, 2.6, 1.2, 0.7, r"$G$", "#FFFFFF", fs=10)

    _box(ax, 1.8, 1.35, 2.0, 0.85, r"$h_{AZ}$ (40D)", COL_STAT)
    _box(ax, 4.0, 0.55, 2.0, 0.85, r"$h_{AG}$ (40D)", COL_STAT)
    _box(ax, 6.2, 1.35, 2.0, 0.85, r"$h_{ZG}$ (40D)", COL_STAT)

    _box(ax, 3.5, 2.95, 3.0, 0.55, r"independent encoders $E_{AZ},E_{AG},E_{ZG}$ : 40→8", COL_FUSE, fs=8)
    _box(ax, 3.8, 3.75, 2.4, 0.55, r"$g_{\mathrm{stat}}\in\mathbb{R}^{24}$", COL_FUSE, fs=9)

    _arrow(ax, 1.6, 2.6, 2.5, 2.2)
    _arrow(ax, 5.0, 2.6, 2.5, 2.2)
    _arrow(ax, 5.0, 2.6, 5.0, 1.45)
    _arrow(ax, 8.4, 2.6, 7.5, 2.2)
    _arrow(ax, 2.8, 1.75, 3.5, 3.0)
    _arrow(ax, 5.0, 1.4, 5.0, 2.95)
    _arrow(ax, 7.2, 1.75, 6.5, 3.0)
    _arrow(ax, 5.0, 3.5, 5.0, 3.75)
    _save(fig, "fig_taskA_motif")


def fig_taskA_architecture() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.text(5, 2.95, "Fusion88 decoder", ha="center", fontsize=11, fontweight="bold")

    _box(ax, 0.3, 1.5, 1.6, 0.9, r"$g_{\mathrm{graph}}$" + "\n64D", COL_GRAPH)
    _box(ax, 0.3, 0.35, 1.6, 0.9, r"$g_{\mathrm{stat}}$" + "\n24D", COL_STAT)
    _box(ax, 2.3, 0.85, 1.5, 1.0, "concat\n88D", COL_FUSE)
    _box(ax, 4.2, 0.85, 1.6, 1.0, "MLP\n88→64", COL_FUSE)
    _box(ax, 6.2, 0.85, 1.4, 1.0, "logit\n64→1", "#FFFFFF")

    _arrow(ax, 1.9, 1.95, 2.25, 1.45)
    _arrow(ax, 1.9, 0.8, 2.25, 1.15)
    _arrow(ax, 3.8, 1.35, 4.15, 1.35)
    _arrow(ax, 5.8, 1.35, 6.15, 1.35)
    _save(fig, "fig_taskA_architecture")


def fig_lastfm_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(5, 4.65, "Last-FM* — frozen pipeline", ha="center", fontsize=11, fontweight="bold")

    _box(ax, 0.15, 3.45, 1.5, 0.75, "CKG", COL_GRAPH)
    _box(ax, 1.85, 3.3, 1.6, 1.0, "HGT\n64D, L2, H2", COL_GRAPH)
    _box(ax, 3.65, 3.3, 1.9, 1.0, r"$q_{uX}$ + dot" + "\n257D", COL_GRAPH)
    _box(ax, 5.75, 3.3, 1.5, 1.0, r"$s_{B0}$ path", COL_FUSE)

    _box(ax, 0.15, 1.55, 1.6, 0.85, r"$H_u$, item $X$", COL_STAT)
    _box(ax, 1.95, 1.55, 1.5, 0.85, "A5 (5D)", COL_STAT)
    _box(ax, 3.65, 1.55, 1.8, 0.85, r"Top25 $\to$ H3", COL_STAT)
    _box(ax, 5.65, 1.55, 1.6, 0.85, "LEG$_{K2}$", COL_STAT)

    _box(ax, 7.5, 2.55, 2.1, 1.05, "Late fusion\n265D MLP", COL_FUSE)
    _box(ax, 7.85, 0.55, 1.4, 0.7, r"$s(u,X)$", "#FFFFFF")

    for x in (1.65, 3.45, 5.55, 7.45):
        _arrow(ax, x, 3.8, x + 0.15, 3.8)
    _arrow(ax, 1.75, 1.95, 1.92, 1.95)
    _arrow(ax, 3.45, 1.95, 3.62, 1.95)
    _arrow(ax, 5.45, 1.95, 5.62, 1.95)
    _arrow(ax, 6.25, 3.8, 7.45, 3.2)
    _arrow(ax, 6.25, 1.95, 7.45, 2.9)
    _arrow(ax, 8.55, 2.55, 8.55, 1.3)
    _save(fig, "fig_lastfm_pipeline")


def fig_lastfm_context() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(5, 3.9, "Adaptive local context for candidate pair (u, X)", ha="center", fontsize=11, fontweight="bold")

    _box(ax, 0.4, 2.5, 1.8, 0.8, r"History $H_u$", COL_STAT)
    _box(ax, 0.4, 1.2, 1.8, 0.8, r"Neighbourhood $N_X$", COL_STAT)
    _box(ax, 7.0, 2.5, 2.2, 0.8, r"Top25$(u,X)$ via $\phi(h,X)$", COL_STAT)
    _box(ax, 7.0, 1.2, 1.5, 0.8, "A5 (5D)", COL_STAT)
    _box(ax, 7.0, 0.15, 1.5, 0.75, "H3 (3D)", COL_STAT)
    _box(ax, 8.7, 0.15, 1.0, 0.75, "LEG", COL_STAT)

    _box(ax, 3.0, 2.05, 2.8, 0.9, r"A11 routing: rank $h\in H_u$ by $\phi(h,X)$", COL_FUSE, fs=8)

    _arrow(ax, 2.2, 2.9, 2.95, 2.5)
    _arrow(ax, 5.8, 2.5, 6.95, 2.5)
    _arrow(ax, 2.2, 1.6, 2.95, 1.6)
    _arrow(ax, 5.8, 1.6, 6.95, 1.6)
    _arrow(ax, 8.1, 2.5, 8.1, 2.05)
    _arrow(ax, 7.75, 1.2, 7.75, 0.95)
    _arrow(ax, 8.5, 0.55, 8.65, 0.55)
    _save(fig, "fig_lastfm_context")


def fig_lastfm_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.text(5, 2.95, "Late fusion + LEG residual", ha="center", fontsize=11, fontweight="bold")

    _box(ax, 0.2, 1.0, 2.0, 0.9, r"$q_{uX}$+dot+A5+H3" + "\n265D", COL_FUSE)
    _box(ax, 2.5, 1.0, 1.5, 0.9, r"$s_{B0}$" + "\nMLP", COL_FUSE)
    _box(ax, 0.2, 0.05, 1.4, 0.75, "LEG$_{K2}$", COL_STAT)
    _box(ax, 2.0, 0.05, 1.5, 0.75, r"$\delta_{\mathrm{LEG}}$", COL_STAT)
    _box(ax, 4.4, 0.85, 1.3, 0.9, r"$+$", "#FFFFFF", fs=12)
    _box(ax, 6.0, 0.85, 1.5, 0.9, r"$s(u,X)$", "#FFFFFF")

    _arrow(ax, 2.2, 1.45, 2.45, 1.45)
    _arrow(ax, 1.6, 0.8, 2.0, 0.55)
    _arrow(ax, 3.5, 0.55, 4.35, 1.15)
    _arrow(ax, 4.0, 1.45, 4.35, 1.3)
    _arrow(ax, 5.7, 1.3, 5.95, 1.3)
    _save(fig, "fig_lastfm_architecture")


def main() -> None:
    _style()
    fig_taskA_pipeline()
    fig_taskA_motif()
    fig_taskA_architecture()
    fig_lastfm_pipeline()
    fig_lastfm_context()
    fig_lastfm_architecture()
    print(f"Wrote architecture figures to {OUT}")


if __name__ == "__main__":
    main()
