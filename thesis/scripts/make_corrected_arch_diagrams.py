"""Corrected Task A and Last-FM* architecture overview diagrams for the thesis.

Implements the scientific-narrative corrections:
  Task A: patient-statistics vs edge candidate split; xv=[μ,σ,m]; mask absent roles
  Last-FM: A5 from Hu+X+NX; Top25 → H3 and LEGK2; ∥ concat vs ⊕ residual; φ legend
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "figures"

# Navy = learned graph; teal = empirical; green = train/eval
# Motif-matched pastels (Fig. 5.2 TikZ: orange!12 / teal!10 / green!12)
NAVY = "#1e3a5f"
NAVY_FILL = "#d6e4f0"
TEAL = "#0f766e"
TEAL_FILL = "#e6f5f4"
ORANGE = "#c2410c"
ORANGE_FILL = "#ffe8d6"
GREEN = "#15803d"
GREEN_FILL = "#e8f5e9"
EDGE = "#2d3748"
FUSE_FILL = "#e8f5e9"
WHITE = "#ffffff"


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


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def _box(ax, x, y, w, h, text, *, ec, fc, fs=7.5, lw=1.15, radius=0.08):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
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
    return (x + w / 2, y + h / 2, x, y, w, h)


def _arrow(ax, p1, p2, *, color=EDGE, lw=1.0):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def _stage_header(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=8.5, fontweight="bold", color=EDGE)


def fig_taskA() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 7.25))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14.2)
    ax.axis("off")
    ax.set_title(
        "Solution architecture",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Stage labels
    _stage_header(ax, 0.3, 13.55, "1  Shared dataset and two data levels")
    _stage_header(ax, 0.3, 8.55, "2  Two-branch architecture")
    _stage_header(ax, 0.3, 3.55, "3  Fusion and prediction")
    _stage_header(ax, 11.8, 3.55, "4  Training and evaluation")

    # --- Stage 1 ---
    _box(ax, 0.4, 11.15, 4.0, 1.25, "Shared synthetic\ndataset", ec=EDGE, fc=WHITE, fs=8)

    # Patient branch: lower under the stage title; fan-out spaced vertically
    _box(ax, 5.0, 12.1, 2.9, 0.72, "Patient records", ec=TEAL, fc=TEAL_FILL, fs=7.2)
    _box(ax, 8.3, 12.1, 3.3, 0.72, "train-patient statistics\n(only)", ec=TEAL, fc=TEAL_FILL, fs=6.5)
    _box(
        ax,
        12.2,
        12.3,
        3.5,
        0.58,
        r"node features $x_v{=}[\mu,\sigma,m]$",
        ec=NAVY,
        fc=NAVY_FILL,
        fs=6.2,
    )
    _box(ax, 12.2, 11.3, 3.7, 0.58, "train-patient pair statistics", ec=ORANGE, fc=ORANGE_FILL, fs=6.2)

    # Candidate branch below patient fan-out
    _box(ax, 5.0, 10.3, 3.5, 0.72, "Candidate-edge table", ec=NAVY, fc=NAVY_FILL, fs=7.0)
    _box(ax, 8.9, 10.3, 3.6, 0.72, "edge-level train/val/test", ec=NAVY, fc=NAVY_FILL, fs=6.7)
    _box(
        ax,
        12.9,
        10.3,
        3.6,
        0.72,
        r"candidate $A\!\to\!G$ (+ context $Z$)",
        ec=NAVY,
        fc=NAVY_FILL,
        fs=6.5,
    )

    _arrow(ax, (4.4, 11.95), (5.0, 12.45), color=TEAL)
    _arrow(ax, (4.4, 11.45), (5.0, 10.65), color=NAVY)
    _arrow(ax, (7.9, 12.45), (8.3, 12.45), color=TEAL)
    _arrow(ax, (11.6, 12.55), (12.2, 12.55), color=NAVY)
    _arrow(ax, (11.6, 12.2), (12.2, 11.55), color=TEAL)
    _arrow(ax, (8.5, 10.65), (8.9, 10.65), color=NAVY)
    _arrow(ax, (12.5, 10.65), (12.9, 10.65), color=NAVY)

    ax.text(
        17.0,
        11.5,
        "Two splits:\n• patients → features/stats\n• edges → labels",
        fontsize=6.8,
        color="#4a5568",
        va="center",
    )

    # --- Stage 2 ---
    # Structural
    ax.add_patch(
        Rectangle((0.35, 4.0), 10.5, 4.3, linewidth=1.0, edgecolor=NAVY, facecolor="#f7fafc", linestyle="--")
    )
    ax.text(0.55, 8.05, "Structural branch (graph context)", fontsize=8, fontweight="bold", color=NAVY)

    _box(ax, 0.5, 6.55, 2.2, 1.15, r"$G_{\mathrm{train}}$" + "\n(+ node feats)", ec=NAVY, fc=NAVY_FILL, fs=6.8)
    _box(
        ax,
        2.95,
        6.4,
        2.75,
        1.45,
        "HGT encoder\n$d{=}32$\n$L{=}2$\n$H{=}4$",
        ec=TEAL,
        fc=TEAL_FILL,
        fs=5.7,
    )
    _box(ax, 5.95, 6.55, 2.2, 1.15, r"$z_A,z_G$" + "\n→ pair $q_{AG}$ 128D", ec=ORANGE, fc=ORANGE_FILL, fs=6.3)
    _box(ax, 8.4, 6.55, 2.15, 1.15, r"$g_{\mathrm{graph}}$" + "\n64D", ec=GREEN, fc=GREEN_FILL, fs=7.0)

    _arrow(ax, (2.7, 7.1), (2.95, 7.1), color=TEAL)
    _arrow(ax, (5.7, 7.1), (5.95, 7.1), color=ORANGE)
    _arrow(ax, (8.15, 7.1), (8.4, 7.1), color=GREEN)

    # Statistical (motif palette: orange = 40D roles, teal = encoders, green = g_stat)
    ax.add_patch(
        Rectangle((11.1, 4.0), 10.5, 4.3, linewidth=1.0, edgecolor=TEAL, facecolor="#f3faf9", linestyle="--")
    )
    ax.text(11.3, 8.05, "Statistical branch (empirical context)", fontsize=8, fontweight="bold", color=TEAL)

    _box(
        ax,
        11.3,
        6.85,
        2.8,
        0.85,
        "train-patient\npair statistics",
        ec=ORANGE,
        fc=ORANGE_FILL,
        fs=7,
    )
    _box(ax, 14.4, 7.35, 1.55, 0.65, "AZ 40D", ec=ORANGE, fc=ORANGE_FILL, fs=7)
    _box(ax, 14.4, 6.55, 1.55, 0.65, "AG 40D", ec=ORANGE, fc=ORANGE_FILL, fs=7)
    _box(ax, 14.4, 5.75, 1.55, 0.65, "ZG 40D", ec=ORANGE, fc=ORANGE_FILL, fs=7)
    _box(
        ax,
        16.3,
        6.2,
        2.6,
        1.4,
        "separate encoders\nfor AZ / AG / ZG\n40→8 each",
        ec=TEAL,
        fc=TEAL_FILL,
        fs=7,
    )
    _box(
        ax,
        19.15,
        6.55,
        2.15,
        1.0,
        "mask absent\nmotif roles",
        ec=TEAL,
        fc=TEAL_FILL,
        fs=7,
    )
    _box(ax, 19.15, 5.2, 2.15, 0.95, r"$g_{\mathrm{stat}}$" + "\n24D", ec=GREEN, fc=GREEN_FILL, fs=8)

    _arrow(ax, (14.1, 7.25), (14.4, 7.65), color=ORANGE)
    _arrow(ax, (14.1, 7.15), (14.4, 6.85), color=ORANGE)
    _arrow(ax, (14.1, 7.05), (14.4, 6.05), color=ORANGE)
    _arrow(ax, (15.95, 7.65), (16.3, 7.2), color=TEAL)
    _arrow(ax, (15.95, 6.85), (16.3, 6.95), color=TEAL)
    _arrow(ax, (15.95, 6.05), (16.3, 6.5), color=TEAL)
    _arrow(ax, (18.9, 6.9), (19.15, 7.05), color=TEAL)
    _arrow(ax, (20.2, 6.55), (20.2, 6.15), color=GREEN)

    # Implicit feed from stage 1 (no long crossing arrows)

    # --- Stage 3 fusion (green like motif g_stat / concat) ---
    _box(
        ax,
        0.5,
        1.35,
        3.6,
        1.7,
        r"$[g_{\mathrm{graph}} \,\Vert\, g_{\mathrm{stat}}]$" + "\n88D",
        ec=GREEN,
        fc=GREEN_FILL,
        fs=8.5,
    )
    _box(ax, 4.5, 1.55, 2.4, 1.3, "MLP\n88→64→1", ec=GREEN, fc=GREEN_FILL, fs=8)
    _box(ax, 7.3, 1.55, 2.0, 1.3, r"edge logit $\ell_{AG}$", ec=EDGE, fc=WHITE, fs=7.5)
    _box(ax, 9.7, 1.55, 2.0, 1.3, r"$\hat{y}_{AG}=\sigma(\ell_{AG})$", ec=EDGE, fc=WHITE, fs=7.5)

    _arrow(ax, (9.5, 6.9), (2.0, 3.1), color=NAVY)
    _arrow(ax, (20.2, 5.2), (2.8, 3.1), color=GREEN)
    _arrow(ax, (4.1, 2.2), (4.5, 2.2), color=GREEN)
    _arrow(ax, (6.9, 2.2), (7.3, 2.2), color=EDGE)
    _arrow(ax, (9.3, 2.2), (9.7, 2.2), color=EDGE)

    # --- Stage 4 ---
    _box(
        ax,
        12.2,
        1.85,
        4.0,
        1.2,
        "3:1 negatives\nweighted BCE",
        ec=GREEN,
        fc=GREEN_FILL,
        fs=7.5,
    )
    _box(
        ax,
        16.5,
        1.85,
        2.4,
        1.2,
        "metric: AP\n(edge-held-out)",
        ec=GREEN,
        fc=GREEN_FILL,
        fs=7.2,
    )
    _box(
        ax,
        12.2,
        0.35,
        9.0,
        1.2,
        "Scenarios: clean · selection bias · no overlap · noisy documentation ·\n"
        "multihospital · hidden confounder†\n"
        "† reported separately; excluded from the official macro",
        ec=GREEN,
        fc=GREEN_FILL,
        fs=6.6,
    )

    # Legend
    ax.text(
        0.4,
        0.15,
        r"Notation:  $\Vert$ = concatenation ·  $\odot$ = element-wise product ·  $|\cdot|$ = abs. difference ·  "
        r"$m$ = train missingness (not correlation) ·  $\sigma$ = sigmoid",
        fontsize=6.2,
        color="#4a5568",
    )

    _save(fig, "fig_taskA_arch_simplified")


def fig_lastfm() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 7.4))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 15.2)
    ax.axis("off")
    ax.set_title(
        "Last-FM* architecture and evaluation protocol",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    _stage_header(ax, 0.3, 14.7, "1  Inputs")
    _stage_header(ax, 0.3, 12.85, "2  Model")
    _stage_header(ax, 18.3, 12.85, "3  Training / evaluation")

    # Inputs
    _box(ax, 0.4, 13.35, 3.2, 0.95, "User $u$ +\ncandidate $X$", ec=EDGE, fc=WHITE, fs=8)
    _box(ax, 4.0, 13.35, 3.4, 0.95, "CKG\n(model_train + music KG)", ec=NAVY, fc=NAVY_FILL, fs=7.2)
    _box(ax, 7.8, 13.35, 2.8, 0.95, r"History $H_u$", ec=TEAL, fc=TEAL_FILL, fs=8)
    _box(ax, 11.0, 13.35, 3.2, 0.95, r"Neighbourhood $N_X$", ec=TEAL, fc=TEAL_FILL, fs=7.5)

    _arrow(ax, (3.6, 13.8), (4.0, 13.8), color=NAVY)
    _arrow(ax, (3.6, 13.7), (7.8, 13.8), color=TEAL)

    # Structural panel
    ax.add_patch(
        Rectangle((0.35, 7.55), 8.9, 4.95, linewidth=1.05, edgecolor=NAVY, facecolor="#f7fafc", linestyle="--")
    )
    ax.text(0.55, 12.2, "Structural branch", fontsize=8.5, fontweight="bold", color=NAVY)

    _box(ax, 0.55, 10.7, 2.5, 1.0, "HGT\n$d{=}64,L{=}2,H{=}2$", ec=NAVY, fc=NAVY_FILL)
    _box(ax, 3.35, 10.7, 2.6, 1.0, r"$z_u,\,z_X\in\mathbb{R}^{64}$", ec=NAVY, fc=NAVY_FILL, fs=7.5)
    _box(
        ax,
        0.55,
        8.9,
        5.4,
        1.35,
        r"$q_{uX}=[z_u\,\Vert\,z_X\,\Vert\,z_u\odot z_X\,\Vert\,|z_u-z_X|]$" + "\n256D",
        ec=NAVY,
        fc=NAVY_FILL,
        fs=7,
    )
    _box(ax, 6.2, 8.9, 2.7, 1.35, r"$\langle z_u,z_X\rangle$" + "\n1D", ec=NAVY, fc=NAVY_FILL, fs=7.5)
    _box(
        ax,
        0.55,
        7.75,
        8.35,
        0.85,
        r"$[q_{uX}\,\Vert\,\langle z_u,z_X\rangle]$  =  structural 257D",
        ec=NAVY,
        fc=NAVY_FILL,
        fs=8,
    )

    _arrow(ax, (5.7, 13.55), (1.8, 11.75), color=NAVY)
    _arrow(ax, (3.05, 11.2), (3.35, 11.2), color=NAVY)
    _arrow(ax, (4.6, 10.7), (3.2, 10.3), color=NAVY)
    _arrow(ax, (5.0, 10.7), (7.2, 10.3), color=NAVY)
    _arrow(ax, (3.2, 8.9), (3.2, 8.6), color=NAVY)
    _arrow(ax, (7.5, 8.9), (5.5, 8.6), color=NAVY)

    # Context panel
    ax.add_patch(
        Rectangle((9.5, 4.55), 8.4, 7.95, linewidth=1.05, edgecolor=TEAL, facecolor="#f3faf9", linestyle="--")
    )
    ax.text(9.7, 12.2, "Candidate-specific empirical context", fontsize=8.5, fontweight="bold", color=TEAL)

    _box(ax, 9.75, 10.9, 3.5, 0.95, r"$H_u$ + $X$", ec=TEAL, fc=TEAL_FILL, fs=8)
    _box(ax, 13.5, 10.9, 2.0, 0.95, r"$N_X$", ec=TEAL, fc=TEAL_FILL, fs=8)
    _box(
        ax,
        9.75,
        9.55,
        5.75,
        1.0,
        r"A5 (5D): size, pop($X$), KG-deg($X$), $J_{\mathrm{set}},C_{\mathrm{set}}(H_u,N_X)$",
        ec=TEAL,
        fc=TEAL_FILL,
        fs=6.6,
    )

    _arrow(ax, (10.6, 13.35), (11.2, 11.9), color=TEAL)
    _arrow(ax, (12.6, 13.35), (14.4, 11.9), color=TEAL)
    _arrow(ax, (11.5, 10.9), (11.5, 10.55), color=TEAL)
    _arrow(ax, (14.5, 10.9), (13.5, 10.55), color=TEAL)

    # A11 routing → Top25 → split to H3 and LEG
    _box(
        ax,
        9.75,
        8.15,
        5.75,
        1.05,
        r"for $h\in H_u$: cross-fitted $A_{11}/\phi(h,X)$" + "\n→ Top25 selected signed associations",
        ec=TEAL,
        fc=TEAL_FILL,
        fs=7,
    )
    _arrow(ax, (11.5, 9.55), (11.5, 9.25), color=TEAL)

    # Fork
    _box(ax, 9.75, 6.55, 2.4, 1.15, "H3 (3D)\nmean / max / top3", ec=TEAL, fc=TEAL_FILL, fs=7)
    _box(ax, 13.1, 6.55, 2.4, 1.15, "LEG$_{K2}$\n$P_2$ mean → MLP", ec=TEAL, fc=TEAL_FILL, fs=7)
    _box(ax, 13.1, 5.0, 2.4, 1.1, r"$\delta_{\mathrm{LEG}}$" + "\n(residual)", ec=TEAL, fc=TEAL_FILL, fs=7.5)

    _arrow(ax, (11.5, 8.15), (10.9, 7.75), color=TEAL)
    _arrow(ax, (12.8, 8.15), (14.2, 7.75), color=TEAL)
    _arrow(ax, (14.3, 6.55), (14.3, 6.15), color=TEAL)

    ax.text(
        12.6,
        7.95,
        "same Top25",
        ha="center",
        fontsize=6.5,
        color=TEAL,
        fontstyle="italic",
    )

    # Fusion
    _box(
        ax,
        0.5,
        5.35,
        8.6,
        1.55,
        r"$[\,\mathrm{structural}\,257\mathrm{D}\;\Vert\;\mathrm{A5}\,5\mathrm{D}\;\Vert\;\mathrm{H3}\,3\mathrm{D}\,]$"
        + "\n= 265D  →  MLP decoder  →  $s_{B0}$",
        ec=EDGE,
        fc=FUSE_FILL,
        fs=8,
    )
    _arrow(ax, (4.7, 7.75), (4.7, 6.95), color=NAVY)
    _arrow(ax, (12.6, 9.55), (7.5, 6.5), color=TEAL)
    _arrow(ax, (10.9, 6.55), (6.5, 6.2), color=TEAL)

    # Final residual addition
    circ = Circle((5.5, 4.35), 0.38, edgecolor=EDGE, facecolor=WHITE, linewidth=1.2)
    ax.add_patch(circ)
    ax.text(5.5, 4.35, r"$\oplus$", ha="center", va="center", fontsize=11)
    _box(ax, 6.3, 3.85, 3.6, 1.0, r"$s(u,X)=s_{B0}+\delta_{\mathrm{LEG}}$", ec=EDGE, fc=WHITE, fs=8)

    _arrow(ax, (4.7, 5.35), (5.2, 4.7))
    _arrow(ax, (14.3, 5.0), (5.95, 4.45), color=TEAL)
    _arrow(ax, (5.9, 4.35), (6.3, 4.35))

    # Right: train / eval strip
    ax.add_patch(
        Rectangle((18.2, 4.55), 5.4, 7.95, linewidth=1.05, edgecolor=GREEN, facecolor="#f4faf4", linestyle="--")
    )
    ax.text(18.4, 12.2, "Protocol", fontsize=8.5, fontweight="bold", color=GREEN)

    _box(
        ax,
        18.4,
        10.6,
        5.0,
        1.35,
        "TRAINING\npos. + 4 negatives\n→ weighted BCE",
        ec=GREEN,
        fc=GREEN_FILL,
        fs=7.5,
    )
    _box(
        ax,
        18.4,
        8.55,
        5.0,
        1.7,
        "CHECKPOINT SELECTION\nval. pos. + 20 negatives\n→ sampled NDCG@20\n→ best checkpoint",
        ec=GREEN,
        fc=GREEN_FILL,
        fs=7.2,
    )
    _box(
        ax,
        18.4,
        5.9,
        5.0,
        2.25,
        "FULL-CATALOG EVAL\nbest checkpoint\nrank all eligible\n$C_u=I\\setminus H_u^{\\mathrm{train}}$\n→ NDCG@20 / Recall@20 / MRR",
        ec=GREEN,
        fc=GREEN_FILL,
        fs=7,
    )
    _arrow(ax, (20.9, 10.6), (20.9, 10.3), color=GREEN)
    _arrow(ax, (20.9, 8.55), (20.9, 8.2), color=GREEN)

    # Legend
    ax.text(
        0.4,
        3.1,
        "Legend",
        fontsize=8,
        fontweight="bold",
        color=EDGE,
    )
    _box(ax, 0.4, 2.15, 2.6, 0.7, "learned graph", ec=NAVY, fc=NAVY_FILL, fs=7)
    _box(ax, 3.2, 2.15, 2.8, 0.7, "empirical context", ec=TEAL, fc=TEAL_FILL, fs=7)
    _box(ax, 6.2, 2.15, 2.6, 0.7, "train / eval", ec=GREEN, fc=GREEN_FILL, fs=7)

    ax.text(
        0.4,
        1.35,
        r"$\Vert$ = concatenation (not addition) ·  $\oplus$ = residual addition ·  "
        r"$A_{11}/\phi$: cross-fitted item–item association "
        r"(Pearson $\phi$; for binary incidence ≡ $A_{11}$)",
        fontsize=6.4,
        color="#4a5568",
    )
    ax.text(
        0.4,
        0.7,
        r"H3 and LEG$_{K2}$ share the same Top25 routed associations; "
        r"$N_X$ enters A5 (set overlap), not the H3/LEG routing path.",
        fontsize=6.4,
        color="#4a5568",
    )
    ax.text(
        0.4,
        0.2,
        r"Structural 257D $=256+1$; fusion 265D $=257+5+3$; LEG enters only as $\delta_{\mathrm{LEG}}$ outside the 265D concat.",
        fontsize=6.4,
        color="#4a5568",
    )

    _save(fig, "fig_lastfm_arch_simplified")


def main() -> None:
    _style()
    fig_taskA()
    fig_lastfm()


if __name__ == "__main__":
    main()
