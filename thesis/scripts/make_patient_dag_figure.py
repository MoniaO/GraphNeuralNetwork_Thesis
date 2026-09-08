#!/usr/bin/env python3
"""Readable per-patient DAG: paths from this patient's active AKI-ancestors to AKI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures"

DATA_CANDIDATES = [
    Path("/Users/monika/data"),
    Path("/Users/monika/2. Data v3/2 v3. Data-kopia/dataset_v3"),
    ROOT / "2 v3. Data" / "dataset_v3",
]

TYPE_COLORS = {
    "patient_context": "#4C78A8",
    "drug_exposure": "#F58518",
    "mechanism": "#54A24B",
    "adr_or_intermediate_state": "#E45756",
    "observation_or_selection": "#B279A2",
    "clinical_endpoint": "#72B7B2",
}

LAYER_ORDER = [
    "1_patient_context",
    "2_drugs",
    "3_mechanisms",
    "4_intermediate_states",
    "5_observation_selection",
    "6_endpoints",
]

def find_data_dir() -> Path:
    for d in DATA_CANDIDATES:
        if (d / "synthetic_pharmacotherapy_v3_nodes.csv").exists() and (
            d / "synthetic_pharmacotherapy_v3_edges_audited.csv"
        ).exists():
            return d
    raise FileNotFoundError("Could not find SPG v3 node/edge CSVs.")


def find_samples(data_dir: Path) -> Path:
    for p in [
        data_dir / "synthetic_pharmacotherapy_v3_samples_clean.csv",
        Path("/Users/monika/2. Data v3/2 v3. Data-kopia/dataset_v3")
        / "synthetic_pharmacotherapy_v3_samples_clean.csv",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find samples_clean.csv.")


SHORT = {
    "sex_female": "female",
    "baseline_egfr": "eGFR",
    "polypharmacy_burden": "polypharm.",
    "depression_history": "depr. hx",
    "heart_failure": "HF",
    "beta_blocker": "beta-bl.",
    "nsaid": "NSAID",
    "nephrotoxin_load": "nephr. load",
    "nephrotoxic_stress": "nephr. stress",
    "creatinine_rise": "creat. rise",
    "hypertension": "HTN",
    "hypotension": "hypotens.",
}


def node_label(name: str, value: float | None) -> str:
    shown = SHORT.get(name, name.replace("_", "\n"))
    if value is None:
        return shown
    if abs(value - round(value)) < 1e-6:
        return f"{shown}\nx={int(round(value))}"
    return f"{shown}\nx={value:.1f}"


def is_active(name: str, value: float, value_type: str) -> bool:
    if name == "unobserved_severity":
        return False
    if value_type in {"continuous", "count"}:
        return True
    return abs(value) > 0


def build_subgraph(
    patient_id: int = 101,
    endpoint: str = "AKI",
    data_dir: Path | None = None,
    samples_path: Path | None = None,
) -> tuple[nx.DiGraph, set[str], dict[str, float]]:
    data_dir = data_dir or find_data_dir()
    nodes = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v3_nodes.csv")
    edges = pd.read_csv(data_dir / "synthetic_pharmacotherapy_v3_edges_audited.csv")
    samples = pd.read_csv(samples_path or find_samples(data_dir))
    match = samples.loc[samples["patient_id"] == patient_id]
    if match.empty:
        raise ValueError(f"patient_id={patient_id} not found in samples")
    row = match.iloc[0]
    meta = nodes.set_index("node")
    if endpoint not in meta.index:
        raise ValueError(f"Unknown endpoint node: {endpoint}")

    G = nx.DiGraph()
    for n, rec in meta.iterrows():
        G.add_node(str(n), **rec.to_dict())
    for _, rec in edges.iterrows():
        G.add_edge(str(rec["source"]), str(rec["target"]))

    ancestors = nx.ancestors(G, endpoint) | {endpoint}
    active: set[str] = set()
    values: dict[str, float] = {}
    for n in ancestors:
        if n not in row.index:
            continue
        val = float(row[n])
        values[n] = val
        vtype = str(meta.loc[n, "value_type"]) if n in meta.index else "binary"
        if is_active(n, val, vtype):
            active.add(n)

    keep: set[str] = set()
    for src in active:
        try:
            keep |= set(nx.shortest_path(G, src, endpoint))
        except nx.NetworkXNoPath:
            continue

    H = G.subgraph(keep).copy()
    for n in H.nodes:
        H.nodes[n]["active"] = n in active
        H.nodes[n]["value"] = values.get(n)
    return H, active, values


def layered_pos(H: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Major x = SPG layer; within a layer, x-offset follows generations so edges show."""
    buckets: dict[str, list[str]] = {layer: [] for layer in LAYER_ORDER}
    for n, data in H.nodes(data=True):
        layer = str(data.get("layer", "1_patient_context"))
        buckets.setdefault(layer, []).append(n)
    used = [layer for layer in LAYER_ORDER if buckets.get(layer)]
    pos: dict[str, tuple[float, float]] = {}
    for xi, layer in enumerate(used):
        names = buckets[layer]
        sub = H.subgraph(names).copy()
        x_left = xi * 3.7
        if sub.number_of_edges() and nx.is_directed_acyclic_graph(sub):
            generations: list[list[str]] = []
            prev: list[str] = []
            for gen in nx.topological_generations(sub):
                ordered = _order_generation(list(gen), prev, sub)
                generations.append(ordered)
                prev = ordered
        else:
            generations = [sorted(names)]
        n_gen = max(len(generations), 1)
        span = 2.05 if n_gen > 1 else 0.0
        for gi, gen in enumerate(generations):
            for i, name in enumerate(gen):
                if len(gen) == 1:
                    y = 0.95 * (1 if gi % 2 == 0 else -1)
                else:
                    y = (len(gen) - 1) / 2 - i
                x_off = 0.0 if n_gen == 1 else span * gi / (n_gen - 1)
                pos[name] = (x_left + x_off, y * 1.85)
    return pos


def _order_generation(
    gen: list[str],
    prev: list[str],
    sub: nx.DiGraph,
) -> list[str]:
    if not prev:
        return sorted(gen)
    scored = []
    for n in gen:
        idx = [prev.index(p) for p in sub.predecessors(n) if p in prev]
        scored.append((sum(idx) / len(idx) if idx else len(prev), n))
    return [n for _, n in sorted(scored)]


def draw_arrows(
    ax: plt.Axes,
    pos: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    radius: float = 0.42,
) -> None:
    for src, dst in edges:
        p0 = np.asarray(pos[src], dtype=float)
        p1 = np.asarray(pos[dst], dtype=float)
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        direction = vec / length
        start = p0 + direction * radius
        end = p1 - direction * radius
        vertical = abs(p0[0] - p1[0]) < 0.25
        rad = 0.06
        if vertical:
            rad = 0.22 if length < 2.2 else 0.10
        if p0[1] < p1[1]:
            rad = -rad
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                lw=1.25,
                color="#8a8a8a",
                connectionstyle=f"arc3,rad={rad:.2f}",
                shrinkA=0,
                shrinkB=0,
                clip_on=False,
                zorder=1,
            )
        )


def draw(
    patient_id: int = 101,
    endpoint: str = "AKI",
    out: Path | None = None,
    data_dir: Path | None = None,
    samples_path: Path | None = None,
) -> Path:
    H, active, _values = build_subgraph(
        patient_id, endpoint=endpoint, data_dir=data_dir, samples_path=samples_path
    )
    pos = layered_pos(H)
    edge_list = [(s, t) for s, t in H.edges() if s in pos and t in pos]

    fig, ax = plt.subplots(figsize=(13.2, 8.2), dpi=200)
    draw_arrows(ax, pos, edge_list)

    for ntype, color in TYPE_COLORS.items():
        nodelist = [n for n, d in H.nodes(data=True) if d.get("node_type") == ntype]
        if not nodelist:
            continue
        active_nodes = [n for n in nodelist if H.nodes[n].get("active")]
        inactive_nodes = [n for n in nodelist if not H.nodes[n].get("active")]
        if active_nodes:
            nx.draw_networkx_nodes(
                H,
                pos,
                nodelist=active_nodes,
                ax=ax,
                node_color=color,
                node_size=[3200 if n == endpoint else 2500 for n in active_nodes],
                edgecolors="#333333",
                linewidths=1.4,
            )
        if inactive_nodes:
            nx.draw_networkx_nodes(
                H,
                pos,
                nodelist=inactive_nodes,
                ax=ax,
                node_color="white",
                node_size=2300,
                edgecolors=color,
                linewidths=1.6,
            )

    active_labels = {
        n: node_label(n, d.get("value"))
        for n, d in H.nodes(data=True)
        if d.get("active")
    }
    inactive_labels = {
        n: node_label(n, None)
        for n, d in H.nodes(data=True)
        if not d.get("active")
    }
    nx.draw_networkx_labels(
        H, pos, labels=active_labels, font_size=8.0, font_color="white", ax=ax
    )
    nx.draw_networkx_labels(
        H, pos, labels=inactive_labels, font_size=8.0, font_color="#222222", ax=ax
    )

    handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=c,
            markeredgecolor="black", markersize=8,
            label=t.replace("_", " "),
        )
        for t, c in TYPE_COLORS.items()
        if any(d.get("node_type") == t for _, d in H.nodes(data=True))
    ]
    handles.append(
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="white",
            markeredgecolor="black", markersize=8,
            label="inactive mediator (outline)",
        )
    )
    ax.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        fancybox=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    ax.set_title(
        f"NetworkX DAG fragment  |  patient_id={patient_id}  |  endpoint={endpoint}",
        fontsize=10,
        pad=10,
        loc="left",
    )
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.4, max(xs) + 1.4)
    ax.set_ylim(min(ys) - 1.4, max(ys) + 1.5)
    ax.axis("off")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    if out is None and patient_id == 101 and endpoint == "AKI":
        png = OUT / "patient_dag_101.png"
    else:
        png = Path(out) if out else OUT / f"patient_dag_{patient_id}_{endpoint}.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw a layered per-patient DAG fragment from SPG v3 CSVs."
    )
    parser.add_argument("--patient-id", type=int, default=101)
    parser.add_argument("--endpoint", default="AKI")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(
        draw(
            args.patient_id,
            endpoint=args.endpoint,
            out=args.out,
            data_dir=args.data_dir,
            samples_path=args.samples,
        )
    )


if __name__ == "__main__":
    main()
