"""Redraw Last-FM* Fig. 6.3: candidate-conditioned history selection.

Palette matches Task A / motif figures:
  navy   = inputs
  orange = A11/phi routing + Top25 associations
  teal   = A5 whole-history summary
  green  = H3 / LEG outputs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures"

NAVY = "#1e3a5f"
NAVY_FILL = "#d6e4f0"
TEAL = "#0f766e"
TEAL_FILL = "#e6f5f4"
ORANGE = "#c2410c"
ORANGE_FILL = "#ffe8d6"
GREEN = "#15803d"
GREEN_FILL = "#e8f5e9"
EDGE = "#2d3748"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _box(ax, x, y, w, h, text, *, ec, fc, fs=7.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.08",
            linewidth=1.15,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=EDGE,
        linespacing=1.15,
    )
    return x + w / 2, y + h / 2


def _arrow(ax, p1, p2, *, color=EDGE, lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    ax.set_title("Candidate-conditioned history selection", fontsize=12, fontweight="bold", pad=8)

    # --- Inputs (navy) ---
    _box(ax, 0.4, 7.15, 3.0, 1.25, r"History $H_u$" + "\nuser training items", ec=NAVY, fc=NAVY_FILL, fs=8)
    _box(ax, 0.4, 5.35, 3.0, 1.25, r"Candidate $X$" + "\ntarget catalog item", ec=NAVY, fc=NAVY_FILL, fs=8)
    _box(ax, 0.4, 3.55, 3.0, 1.25, r"Neighbourhood $N_X$" + "\nco-occurrence neighbours", ec=NAVY, fc=NAVY_FILL, fs=7.5)

    # --- Routing path (orange) ---
    _box(
        ax,
        4.3,
        6.55,
        4.0,
        1.55,
        r"A11 / $\phi(h,X)$ scoring" + "\ncross-fitted association" + "\nfor each $h\\in H_u$",
        ec=ORANGE,
        fc=ORANGE_FILL,
        fs=7.2,
    )
    _box(ax, 9.0, 6.75, 2.6, 1.15, "Top25\nselected associations", ec=ORANGE, fc=ORANGE_FILL, fs=7.5)

    # --- A5 path (teal): whole history + NX ---
    _box(
        ax,
        4.3,
        3.35,
        4.0,
        1.7,
        "A5 (5D)\n"
        + r"size, pop($X$), KG-deg($X$)"
        + "\n"
        + r"$J_{\mathrm{set}},\,C_{\mathrm{set}}(H_u,N_X)$",
        ec=TEAL,
        fc=TEAL_FILL,
        fs=7.0,
    )

    # --- Outputs from Top25 (green) ---
    _box(ax, 12.3, 7.35, 3.1, 1.15, "H3 (3D)\nmean / max / top3", ec=GREEN, fc=GREEN_FILL, fs=7.5)
    _box(
        ax,
        12.3,
        5.55,
        3.1,
        1.25,
        r"LEG$_{K2}$" + "\n" + r"$P_2$ mean → $\delta_{\mathrm{LEG}}$",
        ec=GREEN,
        fc=GREEN_FILL,
        fs=7.3,
    )

    # Bracket note for shared Top25
    ax.plot([11.75, 11.75], [5.7, 8.35], color=ORANGE, lw=1.2, ls="--")
    ax.plot([11.75, 12.15], [5.7, 5.7], color=ORANGE, lw=1.2, ls="--")
    ax.plot([11.75, 12.15], [8.35, 8.35], color=ORANGE, lw=1.2, ls="--")
    ax.text(
        11.55,
        7.0,
        "same\nrouted\nassoc.",
        ha="right",
        va="center",
        fontsize=6.5,
        color=ORANGE,
        rotation=0,
    )

    # Arrows
    _arrow(ax, (3.4, 7.75), (4.3, 7.5), color=ORANGE)
    _arrow(ax, (3.4, 6.0), (4.3, 7.0), color=ORANGE)
    _arrow(ax, (8.3, 7.3), (9.0, 7.3), color=ORANGE)
    _arrow(ax, (11.6, 7.55), (12.3, 7.9), color=GREEN)
    _arrow(ax, (11.6, 7.05), (12.3, 6.3), color=GREEN)

    _arrow(ax, (3.4, 7.4), (4.3, 4.7), color=TEAL)  # Hu → A5
    _arrow(ax, (3.4, 5.7), (4.3, 4.4), color=TEAL)  # X → A5
    _arrow(ax, (3.4, 4.15), (4.3, 4.0), color=TEAL)  # NX → A5

    # Legend
    ax.add_patch(FancyBboxPatch((0.4, 0.35), 0.45, 0.35, boxstyle="round,pad=0.01,rounding_size=0.05", ec=NAVY, fc=NAVY_FILL, lw=1.0))
    ax.text(1.0, 0.52, "Inputs", fontsize=7, va="center", color=EDGE)
    ax.add_patch(FancyBboxPatch((2.6, 0.35), 0.45, 0.35, boxstyle="round,pad=0.01,rounding_size=0.05", ec=ORANGE, fc=ORANGE_FILL, lw=1.0))
    ax.text(3.2, 0.52, r"A11/$\phi$ routing + Top25", fontsize=7, va="center", color=EDGE)
    ax.add_patch(FancyBboxPatch((7.4, 0.35), 0.45, 0.35, boxstyle="round,pad=0.01,rounding_size=0.05", ec=TEAL, fc=TEAL_FILL, lw=1.0))
    ax.text(8.0, 0.52, "A5 whole-history context", fontsize=7, va="center", color=EDGE)
    ax.add_patch(FancyBboxPatch((12.0, 0.35), 0.45, 0.35, boxstyle="round,pad=0.01,rounding_size=0.05", ec=GREEN, fc=GREEN_FILL, lw=1.0))
    ax.text(12.6, 0.52, r"H3 / LEG$_{K2}$ summaries", fontsize=7, va="center", color=EDGE)

    ax.text(
        8.0,
        1.35,
        r"A5 uses $H_u{+}X{+}N_X$ (no Top25).  H3 and LEG$_{K2}$ share the same Top25 associations; "
        r"LEG enters only as residual $\delta_{\mathrm{LEG}}$.",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#4a5568",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_lastfm_context.pdf")
    fig.savefig(OUT / "fig_lastfm_context.png")
    plt.close(fig)
    print("wrote fig_lastfm_context.pdf/.png")


if __name__ == "__main__":
    main()
