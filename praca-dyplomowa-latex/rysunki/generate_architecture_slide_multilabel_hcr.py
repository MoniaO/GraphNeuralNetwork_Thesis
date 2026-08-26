"""Regenerate architecture_slide_multilabel_hcr.png with a white background."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTPUT_PNG = Path(__file__).with_name("architecture_slide_multilabel_hcr.png")
OUTPUT_PDF = Path(__file__).with_name("architecture_slide_multilabel_hcr.pdf")

STYLE = {
    "cream": {"face": "#f5efe6", "edge": "#8d7b68", "text": "#4a4035"},
    "purple": {"face": "#ebe4f7", "edge": "#5e548e", "text": "#3c2f5a"},
    "green": {"face": "#d4f0e8", "edge": "#2d6a4f", "text": "#1b4332"},
    "orange": {"face": "#fde8d8", "edge": "#c47a4a", "text": "#5c3d24"},
    "dim_purple": {"face": "#ebe4f7", "edge": "#5e548e", "text": "#3c2f5a"},
    "dim_green": {"face": "#d4f0e8", "edge": "#2d6a4f", "text": "#1b4332"},
}


def box(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    style: str = "purple",
    fontsize_title: float = 11.5,
    fontsize_sub: float = 9.5,
) -> tuple[float, float, float, float]:
    s = STYLE[style]
    x = cx - w / 2
    y = cy - h / 2
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.5,
        edgecolor=s["edge"],
        facecolor=s["face"],
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            cx,
            cy + 0.012,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize_title,
            fontweight="bold",
            color=s["text"],
            zorder=3,
        )
        ax.text(
            cx,
            cy - 0.018,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize_sub,
            color=s["text"],
            zorder=3,
        )
    else:
        ax.text(
            cx,
            cy,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize_title,
            fontweight="bold",
            color=s["text"],
            zorder=3,
        )
    return x, y, w, h


def hline(ax, x1: float, y: float, x2: float, color: str = "#555555", lw: float = 1.3) -> None:
    ax.plot([x1, x2], [y, y], color=color, linewidth=lw, transform=ax.transAxes, zorder=1)


def vline(ax, x: float, y1: float, y2: float, color: str = "#555555", lw: float = 1.3) -> None:
    ax.plot([x, x], [y1, y2], color=color, linewidth=lw, transform=ax.transAxes, zorder=1)


def elbow(
    ax,
    points: list[tuple[float, float]],
    color: str = "#555555",
    lw: float = 1.3,
) -> None:
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=lw, transform=ax.transAxes, zorder=1, solid_capstyle="round")


def label(ax, x: float, y: float, text: str, ha: str = "center", rotation: float = 0) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="center",
        fontsize=9,
        color="#444444",
        rotation=rotation,
        zorder=4,
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(13, 10.5), facecolor="white", dpi=150)
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Top input ---
    box(
        ax,
        0.50,
        0.93,
        0.30,
        0.08,
        "Patient HeteroData",
        "5 node types, shared topology",
        "cream",
    )

    # split from top
    vline(ax, 0.50, 0.89, 0.865)
    hline(ax, 0.28, 0.865, 0.72)
    vline(ax, 0.28, 0.865, 0.83)
    vline(ax, 0.72, 0.865, 0.83)

    # --- Left (GNN) path ---
    box(
        ax,
        0.28,
        0.785,
        0.34,
        0.08,
        "Per-type input projection",
        "one Linear per type, ID embedding",
        "purple",
    )
    vline(ax, 0.28, 0.745, 0.705)
    box(
        ax,
        0.28,
        0.665,
        0.34,
        0.08,
        "HeteroConv × L",
        "TransformerConv per relation, residual",
        "purple",
    )

    # z endpoint embeddings (spread so boxes do not overlap)
    z_centers = [0.13, 0.28, 0.43]
    z_w, z_h = 0.078, 0.052
    vline(ax, 0.28, 0.625, 0.585)
    hline(ax, z_centers[0], 0.585, z_centers[-1])
    for cx in z_centers:
        vline(ax, cx, 0.585, 0.555)

    for cx, name in zip(z_centers, ["z(AKI)", "z(DILI)", "z(…)"]):
        box(ax, cx, 0.525, z_w, z_h, name, style="purple", fontsize_title=10)

    hline(ax, z_centers[0], 0.495, z_centers[-1])
    vline(ax, 0.28, 0.495, 0.455)

    box(
        ax,
        0.28,
        0.415,
        0.34,
        0.08,
        "Shared classification head",
        "same MLP for all 10 endpoints",
        "purple",
    )

    # --- Right (HCR) path ---
    box(
        ax,
        0.72,
        0.785,
        0.34,
        0.08,
        "HCR evidence tensor",
        "channels per parent",
        "green",
    )
    vline(ax, 0.72, 0.745, 0.705)
    box(
        ax,
        0.72,
        0.665,
        0.34,
        0.08,
        "Pair encoder",
        "shared or per parent type",
        "green",
    )

    # concatenate -> shared head
    elbow(ax, [(0.72, 0.665), (0.72, 0.415), (0.45, 0.415)], lw=1.4)
    label(ax, 0.60, 0.43, "concatenate", ha="center")

    # add to logit -> logits
    elbow(ax, [(0.72, 0.625), (0.86, 0.625), (0.86, 0.245), (0.62, 0.245)], lw=1.4)
    label(ax, 0.88, 0.44, "add to logit", ha="left", rotation=90)

    # --- Bottom output ---
    elbow(ax, [(0.28, 0.375), (0.28, 0.285), (0.50, 0.285)], lw=1.4)
    vline(ax, 0.50, 0.285, 0.28)
    box(ax, 0.50, 0.245, 0.24, 0.07, "10 logits per patient", style="cream", fontsize_title=11)
    vline(ax, 0.50, 0.21, 0.175)
    box(
        ax,
        0.50,
        0.135,
        0.24,
        0.07,
        "BCEWithLogits",
        "per-endpoint pos_weight, capped",
        "orange",
    )

    # --- Dimension legend ---
    label(ax, 0.50, 0.075, "Concatenation: head input grows from 64 to 80 dimensions", ha="center")
    dim_w, dim_h = 0.068, 0.042
    dim_specs = [
        ("z_graph (64)", "dim_purple"),
        ("e_drug (4)", "dim_green"),
        ("e_mech (4)", "dim_green"),
        ("e_adr (4)", "dim_green"),
        ("e_ctx (4)", "dim_green"),
    ]
    dim_gap = 0.018
    total_w = len(dim_specs) * dim_w + (len(dim_specs) - 1) * dim_gap
    start_x = 0.50 - total_w / 2 + dim_w / 2
    for i, (text, style) in enumerate(dim_specs):
        cx = start_x + i * (dim_w + dim_gap)
        box(ax, cx, 0.035, dim_w, dim_h, text, style=style, fontsize_title=8)

    save_kw = dict(facecolor="white", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(OUTPUT_PDF, format="pdf", **save_kw)
    fig.savefig(OUTPUT_PNG, format="png", dpi=600, **save_kw)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
