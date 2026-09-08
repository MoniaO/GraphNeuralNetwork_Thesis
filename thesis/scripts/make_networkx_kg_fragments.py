#!/usr/bin/env python3
"""Generate NetworkX illustrative subgraphs for the thesis (AKI + Last-FM*)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

THESIS = Path(__file__).resolve().parents[1]
FIG = THESIS / "figures"
GSN_DIR = Path(
    "/Users/martajasiewicz/Desktop/GSN Graphs dysertation 2026/2 v3. Data/dataset_v3"
)
LASTFM_DIR = Path(
    "/Users/martajasiewicz/Desktop/KGAT_KnowledgeGraphs/data/LastFM_star_IntentAwareRS"
)

TYPE_COLORS = {
    "patient_context": "#4C78A8",
    "drug_exposure": "#F58518",
    "mechanism": "#54A24B",
    "adr_or_intermediate_state": "#E45756",
    "observation_or_selection": "#B279A2",
    "clinical_endpoint": "#72B7B2",
    "user": "#4C78A8",
    "item": "#F58518",
    "entity": "#54A24B",
}


def _short(label: str, n: int = 18) -> str:
    s = str(label).replace("_", "\n")
    if len(label) <= n:
        return s
    return str(label)[: n - 1] + "…"


def draw_aki_subgraph() -> Path:
    nodes = pd.read_csv(GSN_DIR / "synthetic_pharmacotherapy_v3_nodes.csv")
    edges = pd.read_csv(GSN_DIR / "synthetic_pharmacotherapy_v3_edges_audited.csv")

    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(r["node"], node_type=r["node_type"])
    for _, r in edges.iterrows():
        G.add_edge(str(r["source"]), str(r["target"]), edge_type=str(r["edge_type"]))

    keep = nx.ancestors(G, "AKI") | {"AKI"}
    H = G.subgraph(keep).copy()

    dist: dict[str, int] = {}
    for n in H.nodes:
        try:
            dist[n] = nx.shortest_path_length(H, n, "AKI")
        except Exception:
            dist[n] = 99
    md = max((d for d in dist.values() if d < 99), default=1)
    layers: dict[int, list[str]] = {}
    for n, d in dist.items():
        layers.setdefault(md - d if d < 99 else 0, []).append(n)
    pos: dict[str, tuple[float, float]] = {}
    for lx, nodelist in sorted(layers.items()):
        nodelist = sorted(nodelist)
        for i, n in enumerate(nodelist):
            pos[n] = (lx * 1.8, (len(nodelist) - 1) / 2 - i)

    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=200)
    ax.set_title(
        "AKI-related fragment of Synthetic Pharmacotherapy Graph v3 (ancestor subgraph)",
        fontsize=11,
        pad=10,
    )
    for ntype, color in TYPE_COLORS.items():
        nodelist = [n for n, d in H.nodes(data=True) if d.get("node_type") == ntype]
        if not nodelist:
            continue
        nx.draw_networkx_nodes(
            H,
            pos,
            nodelist=nodelist,
            node_color=color,
            node_size=900,
            edgecolors="white",
            linewidths=1.0,
            ax=ax,
        )
    nx.draw_networkx_edges(
        H,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        width=1.0,
        edge_color="#666666",
        connectionstyle="arc3,rad=0.03",
        ax=ax,
        alpha=0.85,
        min_source_margin=12,
        min_target_margin=12,
    )
    labels = {n: _short(n, 20) for n in H.nodes}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=5.5, font_color="white", ax=ax)

    handles = []
    for ntype, color in TYPE_COLORS.items():
        if any(d.get("node_type") == ntype for _, d in H.nodes(data=True)):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=9,
                    label=ntype.replace("_", " "),
                )
            )
    ax.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    ax.text(
        0.01,
        -0.02,
        f"{H.number_of_nodes()} nodes, {H.number_of_edges()} directed edges "
        "(all ancestors of AKI in the audited v3 graph).",
        transform=ax.transAxes,
        fontsize=7,
        color="#444444",
    )
    ax.axis("off")
    fig.tight_layout()
    out = FIG / "fig_aki_kg_fragment_networkx.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_lastfm_subgraph() -> Path:
    """Small collaborative-knowledge-graph fragment around one user."""
    # Prefer a user with a moderate history so the drawing stays readable.
    train_path = LASTFM_DIR / "train.txt"
    user_items: dict[int, list[int]] = {}
    with train_path.open() as f:
        for line in f:
            parts = [int(x) for x in line.split()]
            if len(parts) < 4:
                continue
            u, items = parts[0], parts[1:]
            if 4 <= len(items) <= 12:
                user_items[u] = items
            if len(user_items) > 2000:
                break
    # Stable choice: smallest user id among candidates with enough KG links later.
    if not user_items:
        raise RuntimeError("No suitable Last-FM* user found for illustration")

    # Load a sample of KG triples involving catalog items (head or tail < n_items).
    n_items = 48123
    item_to_ent: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with (LASTFM_DIR / "kg_final.txt").open() as f:
        for i, line in enumerate(f):
            h, r, t = [int(x) for x in line.split()]
            if h < n_items and t >= n_items:
                item_to_ent[h].append((r, t))
            elif t < n_items and h >= n_items:
                item_to_ent[t].append((r, h))
            if i > 2_000_000:
                break

    # Pick user whose first few items have KG neighbours.
    chosen_u = None
    chosen_items: list[int] = []
    for u in sorted(user_items):
        items = [i for i in user_items[u] if item_to_ent.get(i)]
        if len(items) >= 3:
            chosen_u = u
            chosen_items = items[:4]
            break
    if chosen_u is None:
        chosen_u = min(user_items)
        chosen_items = user_items[chosen_u][:4]

    rel_names = {}
    with (LASTFM_DIR / "relation_list.txt").open() as f:
        next(f)
        for line in f:
            org, rid = line.rsplit(None, 1)
            short = org.strip().split("/")[-1].replace(".", "\n")
            rel_names[int(rid)] = short

    G = nx.DiGraph()
    G.add_node(f"u{chosen_u}", ntype="user", label=f"user\n{chosen_u}")
    for i in chosen_items:
        G.add_node(f"i{i}", ntype="item", label=f"item\n{i}")
        G.add_edge(f"u{chosen_u}", f"i{i}", etype="interact")
        # Attach up to 2 KG neighbours per item.
        for r, e in item_to_ent[i][:2]:
            en = f"e{e}"
            if en not in G:
                G.add_node(en, ntype="entity", label=f"entity\n{e}")
            G.add_edge(f"i{i}", en, etype=rel_names.get(r, f"r{r}"))

    # Layout: user left, items middle, entities right.
    pos = {f"u{chosen_u}": (0.0, 1.5)}
    for k, i in enumerate(chosen_items):
        pos[f"i{i}"] = (2.2, 3.0 - k * 1.0)
    ents = [n for n, d in G.nodes(data=True) if d["ntype"] == "entity"]
    for k, en in enumerate(ents):
        pos[en] = (4.6, 3.2 - k * 0.7)

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=200)
    ax.set_title(
        "Illustrative Last-FM* collaborative knowledge-graph fragment",
        fontsize=11,
        pad=10,
    )
    for ntype, color in [
        ("user", TYPE_COLORS["user"]),
        ("item", TYPE_COLORS["item"]),
        ("entity", TYPE_COLORS["entity"]),
    ]:
        nodelist = [n for n, d in G.nodes(data=True) if d["ntype"] == ntype]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_color=color,
            node_size=1700,
            edgecolors="white",
            linewidths=1.2,
            ax=ax,
        )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=13,
        width=1.3,
        edge_color="#555555",
        connectionstyle="arc3,rad=0.05",
        ax=ax,
        min_source_margin=16,
        min_target_margin=16,
    )
    labels = {n: d["label"] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color="white", ax=ax)
    # Edge labels only for KG relations (not interact), to reduce clutter.
    elabels = {
        (u, v): (d["etype"][:22] + "…") if len(d["etype"]) > 22 else d["etype"]
        for u, v, d in G.edges(data=True)
        if d["etype"] != "interact"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels, font_size=5.5, ax=ax, rotate=False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLORS["user"], markersize=10, label="user"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLORS["item"], markersize=10, label="item (track)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLORS["entity"], markersize=10, label="KG entity"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    ax.text(
        0.02,
        -0.02,
        "Solid edges: user–item interactions from train + Freebase-style KG triples.",
        transform=ax.transAxes,
        fontsize=7,
        color="#444444",
    )
    ax.axis("off")
    fig.tight_layout()
    out = FIG / "fig_lastfm_kg_fragment_networkx.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    aki = draw_aki_subgraph()
    last = draw_lastfm_subgraph()
    print(f"Wrote {aki}")
    print(f"Wrote {last}")


if __name__ == "__main__":
    main()
