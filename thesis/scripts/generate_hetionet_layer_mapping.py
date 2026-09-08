"""Regenerate hetionet layer-mapping figure (PDF + PNG, white background)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_PNG = OUT_DIR / "hetionet_layer_mapping.png"
OUTPUT_PDF = OUT_DIR / "hetionet_layer_mapping.pdf"

LAYERS = [
    {
        "num": 1,
        "slot": "patient_context",
        "entity": "Pharmacologic Class",
        "face": "#d4f0e8",
        "edge": "#2d6a4f",
        "text": "#1b4332",
        "y": 0.82,
    },
    {
        "num": 2,
        "slot": "drug_exposure",
        "entity": "Gene, bound (CbG)",
        "face": "#d4f0e8",
        "edge": "#2d6a4f",
        "text": "#1b4332",
        "y": 0.64,
    },
    {
        "num": 3,
        "slot": "mechanism",
        "entity": "Pathway, Biological Process",
        "face": "#ebe4f7",
        "edge": "#5e548e",
        "text": "#3c2f5a",
        "y": 0.46,
    },
    {
        "num": 4,
        "slot": "adr_or_intermediate_state",
        "entity": "Gene, disease-associated (DaG)",
        "face": "#ebe4f7",
        "edge": "#5e548e",
        "text": "#3c2f5a",
        "y": 0.28,
    },
    {
        "num": 5,
        "slot": "clinical_endpoint",
        "entity": "Disease (CtD, CpD)",
        "face": "#f5efe6",
        "edge": "#8d7b68",
        "text": "#4a4035",
        "y": 0.10,
    },
]

BOX_X = 0.28
BOX_W = 0.52
BOX_H = 0.11


def draw_box(ax, layer: dict) -> tuple[float, float, float, float]:
    y = layer["y"]
    patch = FancyBboxPatch(
        (BOX_X, y),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=layer["edge"],
        facecolor=layer["face"],
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)

    cx = BOX_X + BOX_W / 2
    cy = y + BOX_H / 2
    ax.text(
        cx,
        cy + 0.018,
        f"{layer['num']} · {layer['slot']}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=layer["text"],
        zorder=3,
    )
    ax.text(
        cx,
        cy - 0.022,
        layer["entity"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=layer["text"],
        zorder=3,
    )
    return BOX_X, y, BOX_W, BOX_H


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 12), facecolor="white", dpi=150)
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0.17, 0.17], [0.06, 0.95], color="#444444", linewidth=1.2, transform=ax.transAxes, zorder=1)
    for label, y in [("observed", 0.73), ("latent", 0.37), ("target", 0.10)]:
        ax.text(
            0.08,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=12,
            color="#333333",
            style="italic",
        )

    boxes = [draw_box(ax, layer) for layer in LAYERS]

    def box_bottom(idx: int) -> tuple[float, float]:
        x, y, w, h = boxes[idx]
        return x + w / 2, y

    def box_top(idx: int) -> tuple[float, float]:
        x, y, w, h = boxes[idx]
        return x + w / 2, y + h

    def box_right_mid(idx: int) -> tuple[float, float]:
        x, y, w, h = boxes[idx]
        return x + w, y + h / 2

    edges = [
        (0, 1, "PCiC ◦ CbG (derived, train split)"),
        (1, 2, "GpPW, GpBP"),
        (2, 3, "GpPW, GpBP"),
        (3, 4, "DaG"),
    ]
    for src, dst, label in edges:
        x1, y1 = box_bottom(src)
        x2, y2 = box_top(dst)
        ax.plot([x1, x2], [y1, y2], color="#555555", linewidth=1.3, transform=ax.transAxes, zorder=1)
        ax.text(
            x1 + 0.012,
            (y1 + y2) / 2,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            color="#333333",
        )

    x2, y2 = box_right_mid(1)
    x5, y5 = box_right_mid(4)
    bypass_x = 0.86
    ax.plot(
        [x2, bypass_x, bypass_x, x5 + 0.02],
        [y2, y2, y5, y5],
        color="#555555",
        linewidth=1.3,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.text(
        bypass_x + 0.012,
        (y2 + y5) / 2,
        "DaG direct",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        color="#333333",
        rotation=90,
    )

    save_kw = dict(facecolor="white", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUTPUT_PDF, format="pdf", **save_kw)
    fig.savefig(OUTPUT_PNG, format="png", dpi=600, **save_kw)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
