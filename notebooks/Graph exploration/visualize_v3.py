#!/usr/bin/env python3
"""Create an interactive, filterable HTML visualization of the v3 DAG."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsn_paths import v3_data_dir  # noqa: E402

DATA = v3_data_dir()
DEFAULT_OUTPUT = DATA / "synthetic_pharmacotherapy_v3_dag.html"

COLORS = {
    "patient_context": "#95a5a6",
    "drug_exposure": "#3498db",
    "mechanism": "#9b59b6",
    "adr_or_intermediate_state": "#f39c12",
    "observation_or_selection": "#1abc9c",
    "clinical_endpoint": "#e74c3c",
}
SIZES = {
    "patient_context": 15,
    "drug_exposure": 19,
    "mechanism": 19,
    "adr_or_intermediate_state": 20,
    "observation_or_selection": 16,
    "clinical_endpoint": 28,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def text(value: object, fallback: str = "brak") -> str:
    return str(value) if pd.notna(value) and str(value).strip() else fallback


def main() -> None:
    args = parse_args()
    nodes = pd.read_csv(DATA / "synthetic_pharmacotherapy_v3_nodes.csv")
    edges = pd.read_csv(DATA / "synthetic_pharmacotherapy_v3_edges_audited.csv")
    graph = nx.from_pandas_edgelist(
        edges, "source", "target", edge_attr=True, create_using=nx.DiGraph
    )
    graph.add_nodes_from(nodes["node"])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The v3 graph is not a DAG.")

    metadata = nodes.set_index("node").to_dict(orient="index")
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#000000",
        select_menu=True,
        filter_menu=True,
        cdn_resources="in_line",
        notebook=False,
    )

    for name in nodes["node"]:
        record = metadata[name]
        node_type = str(record["node_type"])
        parents = sorted(graph.predecessors(name))
        children = sorted(graph.successors(name))
        tooltip = (
            f"<b>{name}</b><br>"
            f"Typ: {node_type}<br>"
            f"Warstwa: {text(record.get('layer'))}<br>"
            f"Opis: {text(record.get('description'))}<br>"
            f"Prewalencja bazowa: {text(record.get('base_prevalence'))}<br>"
            f"Rodzice: {len(parents)} | Dzieci: {len(children)}"
            f"<br><br><b>Rodzice:</b> {', '.join(parents) or 'brak'}"
            f"<br><br><b>Dzieci:</b> {', '.join(children) or 'brak'}"
        )
        net.add_node(
            name,
            label=name,
            title=tooltip,
            color=COLORS.get(node_type, "#95a5a6"),
            size=SIZES.get(node_type, 15),
            shape="dot",
            node_type=node_type,
            layer=text(record.get("layer")),
            introduced_in=text(record.get("introduced_in")),
            is_endpoint="yes" if bool(record.get("is_endpoint")) else "no",
            parent_count=str(len(parents)),
            child_count=str(len(children)),
            total_degree=str(graph.degree(name)),
        )

    for row in edges.itertuples(index=False):
        source_type = str(metadata[row.source]["node_type"])
        target_type = str(metadata[row.target]["node_type"])
        tooltip = (
            f"<b>{row.source} → {row.target}</b><br>"
            f"Typ krawędzi: {text(row.edge_type)}<br>"
            f"Znak efektu: {text(row.effect_sign)}<br>"
            f"Siła efektu: {text(row.effect_size)}<br>"
            f"Ścieżka: {text(row.clinical_pathway)}<br>"
            f"Motyw DAG: {text(row.dag_motif)}<br>"
            f"Uzasadnienie: {text(row.rationale)}<br>"
            f"Wersja: {text(row.introduced_in)}"
        )
        net.add_edge(
            row.source,
            row.target,
            title=tooltip,
            arrows="to",
            edge_type=text(row.edge_type),
            clinical_pathway=text(row.clinical_pathway),
            introduced_in=text(row.introduced_in),
            source_type=source_type,
            target_type=target_type,
        )

    net.repulsion(
        node_distance=250,
        central_gravity=0.10,
        spring_length=200,
        spring_strength=0.03,
        damping=0.12,
    )
    net.show_buttons(filter_=["physics"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(args.output), open_browser=False, notebook=False)
    print(f"Interaktywny graf v3 zapisano w:\n{args.output}")
    print(f"Węzły: {len(nodes)} | Krawędzie: {len(edges)} | DAG: tak")
    if args.open_browser and not webbrowser.open(args.output.resolve().as_uri()):
        print("Nie udało się automatycznie otworzyć przeglądarki.")


if __name__ == "__main__":
    main()
