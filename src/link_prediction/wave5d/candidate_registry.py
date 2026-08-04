"""Frozen candidate registry shared by all Wave 5D variants."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from hcr.motif_registry_v3 import DUAL_GATES, GATE_OUTCOMES, TRIPLE_GATES
from hcr.motifs import GATE_DEFINITIONS


GATE_NODES = (
    set(DUAL_GATES)
    | set(TRIPLE_GATES)
    | set(GATE_DEFINITIONS)
)
ENDPOINT_NODES = set(GATE_OUTCOMES)


def _infer_negative_role(
    source: str,
    target: str,
    label: float,
    edge_type: str,
    g_train: nx.DiGraph | None,
    node_type: dict[str, str],
) -> str:
    if float(label) >= 0.5:
        if source in GATE_NODES or target in GATE_NODES:
            return "positive_gate"
        return "positive"
    # Negatives
    et = str(edge_type).lower()
    if et and et != "negative":
        return f"type_matched_{et}"
    st = node_type.get(source, "")
    tt = node_type.get(target, "")
    if tt in {"observation_or_selection"} or st in {"observation_or_selection"}:
        return "descendant_post_outcome"
    if target in ENDPOINT_NODES and source in ENDPOINT_NODES:
        return "collider_nonedge"
    if g_train is not None and source in g_train and target in g_train:
        # Common child → collider-ish; common parent → confounded.
        if set(g_train.successors(source)) & set(g_train.successors(target)):
            return "confounded_nonedge"
        if set(g_train.predecessors(source)) & set(g_train.predecessors(target)):
            return "confounded_nonedge"
        # Mediator: path source→…→target of length ≥2 in train
        if nx.has_path(g_train, source, target):
            try:
                if nx.shortest_path_length(g_train, source, target) >= 2:
                    return "mediator_nonedge"
            except nx.NetworkXNoPath:
                pass
    if st == tt:
        return "semantic_type_matched"
    return "easy_negative"


def build_frozen_candidates(
    evidence: pd.DataFrame,
    *,
    g_train: nx.DiGraph | None = None,
    node_metadata: pd.DataFrame | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Freeze candidates from Wave-5 evidence (identical for all LP variants)."""
    df = evidence.copy()
    required = {
        "candidate_id",
        "source",
        "target",
        "label",
        "edge_type",
        "edge_split",
        "topological_allowed",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Evidence missing columns: {sorted(missing)}")

    node_type: dict[str, str] = {}
    if node_metadata is not None:
        col = "node" if "node" in node_metadata.columns else "node_id"
        tcol = "node_type" if "node_type" in node_metadata.columns else "type"
        node_type = {
            str(r[col]): str(r[tcol])
            for _, r in node_metadata.iterrows()
        }

    roles = [
        _infer_negative_role(
            str(r.source),
            str(r.target),
            float(r.label),
            str(r.edge_type),
            g_train,
            node_type,
        )
        for r in df.itertuples(index=False)
    ]
    out = pd.DataFrame(
        {
            "candidate_id": df["candidate_id"].astype(str),
            "source": df["source"].astype(str),
            "target": df["target"].astype(str),
            "label": df["label"].astype(float),
            "edge_type": df["edge_type"].astype(str),
            "negative_role": roles,
            "mask_split": df["edge_split"].astype(str),
            "topological_allowed": df["topological_allowed"].astype(bool),
            "edge_in_train": df["edge_in_train"].astype(bool)
            if "edge_in_train" in df.columns
            else False,
        }
    )
    if seed is not None:
        out.insert(0, "seed", int(seed))
    return out


def save_frozen_candidates(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
