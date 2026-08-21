"""Frozen candidate↔co-parent triangle used by S10 attach.

Honest motif name (matches frozen L2 structural attach):

    AZ ⊕ AG ⊕ ZG

where:
  A = candidate source
  G = candidate target (gate / mechanism node)
  Z = selected co-parent / context (NOT a downstream child Y)

This is deliberately *not* AB⊕AY⊕BY with Y = GATE_OUTCOMES[G].
Keep the triangle fixed so only the pair encoding changes across S5–S10.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

MOTIF_TYPE = "candidate_coparent_triangle"
PAIR_ROLES = ("AZ", "AG", "ZG")
CONTEXT_SELECTION_RULE = "lexicographic_first_of_sorted_candidates"


def select_context(ctxs: tuple[str, ...] | list[str]) -> tuple[str, tuple[str, ...]]:
    """Deterministic Z = sorted(candidates)[0]."""
    candidates = tuple(sorted({str(c) for c in ctxs if str(c)}))
    if not candidates:
        raise ValueError("empty context candidates")
    return candidates[0], candidates


def build_candidate_coparent_triples(
    context_map: Mapping[tuple[str, str], tuple[str, ...]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Map candidate edge (A,G) → motif metadata and ordered pair roles."""
    triples: dict[tuple[str, str], dict[str, Any]] = {}
    for (a, g), ctxs in context_map.items():
        if not ctxs:
            continue
        z, candidates = select_context(ctxs)
        a_s, g_s = str(a), str(g)
        if z in {a_s, g_s}:
            continue
        triples[(a_s, g_s)] = {
            "A": a_s,
            "G": g_s,
            "Z": z,
            "Y_alias_of_G": g_s,  # legacy name in older code; Y := G
            "pairs": ((a_s, z), (a_s, g_s), (z, g_s)),  # AZ, AG, ZG
            "pair_roles": list(PAIR_ROLES),
            "motif_type": MOTIF_TYPE,
            "context_candidates": list(candidates),
            "context_rank": 0,
            "context_selection_rule": CONTEXT_SELECTION_RULE,
            "selected_context": z,
        }
    return triples


def triples_fingerprint(triples: Mapping[tuple[str, str], dict[str, Any]]) -> str:
    payload = {
        f"{a}|{g}": {
            "Z": meta["Z"],
            "pairs": [list(p) for p in meta["pairs"]],
            "roles": meta["pair_roles"],
            "motif_type": meta["motif_type"],
        }
        for (a, g), meta in sorted(triples.items())
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

