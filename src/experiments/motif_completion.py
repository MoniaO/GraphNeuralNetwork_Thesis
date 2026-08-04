"""Wave 3B protocol A1 — dual-gate motif completion helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from hcr.motifs import (
    DualGate,
    MotifCompletionTask,
    extract_dual_gates,
    load_truth_graph,
    motif_completion_tasks,
)


def binary_dual_gates(cfg: Any, samples: pd.DataFrame | None = None) -> list[DualGate]:
    _, edges = load_truth_graph(cfg)
    return [g for g in extract_dual_gates(edges, samples) if g.binary_parents]


def _hide_mode(cfg: Any) -> str:
    exp = getattr(cfg, "experiment", None)
    mc = getattr(exp, "motif_completion", None) if exp is not None else None
    return str(getattr(mc, "hide", "parent_a")).strip().lower()


def hide_parent_a_tasks(cfg: Any, samples: pd.DataFrame | None = None) -> list[MotifCompletionTask]:
    """Backward-compatible alias: hide parent_a → gate."""
    return hide_tasks(cfg, samples, force_hide="parent_a")


def hide_tasks(
    cfg: Any,
    samples: pd.DataFrame | None = None,
    force_hide: str | None = None,
) -> list[MotifCompletionTask]:
    """Held-out parent→gate tasks for Wave 3B/4C/4D.

    ``experiment.motif_completion.hide``:
      parent_a — hide A→G (keep B→G)
      parent_b — hide B→G (keep A→G)
      both     — both sides (≈20 positives; for catalog / dual-mask audits)
    """
    mode = (force_hide or _hide_mode(cfg)).strip().lower()
    duals = binary_dual_gates(cfg, samples)
    tasks: list[MotifCompletionTask] = []
    for g in duals:
        if mode in {"parent_a", "a", "both"}:
            tasks.append(
                MotifCompletionTask(
                    gate=g.gate,
                    hidden_parent=g.parent_a,
                    visible_parent=g.parent_b,
                    candidate_source=g.parent_a,
                    candidate_target=g.gate,
                    children=g.children,
                )
            )
        if mode in {"parent_b", "b", "both"}:
            tasks.append(
                MotifCompletionTask(
                    gate=g.gate,
                    hidden_parent=g.parent_b,
                    visible_parent=g.parent_a,
                    candidate_source=g.parent_b,
                    candidate_target=g.gate,
                    children=g.children,
                )
            )
    if not tasks:
        raise ValueError(f"Unknown motif_completion.hide={mode!r}")
    return tasks


def held_out_edge_ids(cfg: Any, samples: pd.DataFrame | None = None) -> list[str]:
    _, edges = load_truth_graph(cfg)
    tasks = hide_tasks(cfg, samples)
    ids: list[str] = []
    for t in tasks:
        match = edges[
            (edges["source"].astype(str) == t.candidate_source)
            & (edges["target"].astype(str) == t.candidate_target)
        ]
        if match.empty:
            raise KeyError(
                f"No audited edge for held-out motif edge "
                f"{t.candidate_source}→{t.candidate_target}"
            )
        ids.append(str(match.iloc[0]["edge_id"]))
    return ids


def oracle_z_map(cfg: Any, samples: pd.DataFrame | None = None) -> dict[tuple[str, str], str]:
    """Map held-out (source, target) → context Z.

    z_mode:
      oracle_coparent (default) — true co-parent B
      random_matched — random other binary node (G5 control)
    """
    tasks = hide_tasks(cfg, samples)
    z_mode = "oracle_coparent"
    exp = getattr(cfg, "experiment", None)
    hcr = getattr(cfg, "hcr", None)
    if hcr is not None:
        z_mode = str(getattr(hcr, "z_mode", z_mode)).strip().lower()
    if z_mode in {"oracle_coparent", "oracle", "coparent"}:
        return {(t.candidate_source, t.candidate_target): t.visible_parent for t in tasks}

    if z_mode in {"random_matched", "random"}:
        # Deterministic fake contexts: pick a binary node ≠ A,B,G.
        from hcr.variable_specs_v3 import VARIABLE_SPECS
        from hcr.variable_spec import VariableType
        import hashlib

        binary_nodes = sorted(
            n
            for n, spec in VARIABLE_SPECS.items()
            if spec.variable_type == VariableType.BINARY
        )
        out: dict[tuple[str, str], str] = {}
        for t in tasks:
            forbidden = {t.candidate_source, t.candidate_target, t.visible_parent, t.gate}
            candidates = [n for n in binary_nodes if n not in forbidden]
            if not candidates:
                out[(t.candidate_source, t.candidate_target)] = t.visible_parent
                continue
            digest = hashlib.sha256(
                f"{t.candidate_source}|{t.candidate_target}|{getattr(hcr, 'shuffle_seed', 20260722)}".encode()
            ).hexdigest()
            idx = int(digest[:8], 16) % len(candidates)
            out[(t.candidate_source, t.candidate_target)] = candidates[idx]
        return out

    raise ValueError(f"Unknown hcr.z_mode={z_mode!r}")


def tasks_frame(cfg: Any, samples: pd.DataFrame | None = None) -> pd.DataFrame:
    tasks = hide_tasks(cfg, samples)
    edge_ids = held_out_edge_ids(cfg, samples)
    rows = []
    for t, eid in zip(tasks, edge_ids):
        row = asdict(t)
        row["held_out_edge_id"] = eid
        rows.append(row)
    return pd.DataFrame(rows)


def motif_completion_enabled(cfg: Any) -> bool:
    exp = getattr(cfg, "experiment", None)
    mc = getattr(exp, "motif_completion", None) if exp is not None else None
    return bool(mc is not None and getattr(mc, "enabled", False))


def latent_gate_enabled(cfg: Any) -> bool:
    exp = getattr(cfg, "experiment", None)
    lg = getattr(exp, "latent_gate", None) if exp is not None else None
    return bool(lg is not None and getattr(lg, "enabled", False))


def latent_triple_map(
    cfg: Any,
    samples: pd.DataFrame | None = None,
) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Map held-out candidate (A, G) → HCR-3 columns (A, Y, B).

    Y is the primary downstream outcome (``GATE_OUTCOMES``), never the gate G.
    z_mode controls B:
      oracle_outcome / oracle_coparent — true co-parent B
      random_matched — random binary ≠ A,B,G,Y
    """
    from hcr.motif_registry_v3 import GATE_OUTCOMES

    tasks = hide_tasks(cfg, samples)
    hcr = getattr(cfg, "hcr", None)
    z_mode = str(getattr(hcr, "z_mode", "oracle_outcome")).strip().lower()

    out: dict[tuple[str, str], tuple[str, str, str]] = {}
    for t in tasks:
        outcome = GATE_OUTCOMES.get(t.gate)
        if outcome is None:
            # Fall back to first audited child if registry missing.
            if not t.children:
                continue
            outcome = t.children[0]
        z = t.visible_parent
        if z_mode in {"random_matched", "random"}:
            from hcr.variable_spec import VariableType
            from hcr.variable_specs_v3 import VARIABLE_SPECS
            from hcr.motif_registry_v3 import DUAL_GATES
            import hashlib

            # Never sample another gate node as fake Z (would trip leakage checks
            # and is not a meaningful matched-random patient variable).
            all_gates = set(GATE_OUTCOMES) | set(DUAL_GATES)
            binary_nodes = sorted(
                n
                for n, spec in VARIABLE_SPECS.items()
                if spec.variable_type == VariableType.BINARY and n not in all_gates
            )
            forbidden = {
                t.candidate_source,
                t.candidate_target,
                t.visible_parent,
                t.gate,
                outcome,
            }
            candidates = [n for n in binary_nodes if n not in forbidden]
            if candidates:
                digest = hashlib.sha256(
                    f"latent|{t.candidate_source}|{t.candidate_target}|"
                    f"{getattr(hcr, 'shuffle_seed', 20260722)}".encode()
                ).hexdigest()
                z = candidates[int(digest[:8], 16) % len(candidates)]
        # (x, y, z) = (A, Y, B) — candidate key remains (A, G)
        out[(t.candidate_source, t.candidate_target)] = (
            t.candidate_source,
            outcome,
            z,
        )
    return out


def latent_context_map(
    cfg: Any,
    samples: pd.DataFrame | None = None,
) -> dict[tuple[str, str], dict[str, str]]:
    """Rich map for L1 pairwise concat: A,B,Y,G per held-out edge."""
    triples = latent_triple_map(cfg, samples)
    rich: dict[tuple[str, str], dict[str, str]] = {}
    for (a, g), (x, y, z) in triples.items():
        rich[(a, g)] = {"A": x, "G": g, "Y": y, "B": z}
    return rich
