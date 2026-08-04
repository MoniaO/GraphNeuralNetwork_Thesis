"""Helpers to collect candidate (source, target) name pairs from HeteroData."""

from __future__ import annotations

from typing import Sequence, Tuple

CandidatePair = Tuple[str, str]


def candidate_pairs_from_data(data) -> list[CandidatePair]:
    sources = list(getattr(data, "candidate_source_name"))
    targets = list(getattr(data, "candidate_target_name"))
    if len(sources) != len(targets):
        raise ValueError(
            "candidate_source_name and candidate_target_name length mismatch: "
            f"{len(sources)} vs {len(targets)}"
        )
    return list(zip(sources, targets))


def unique_pairs(pairs: Sequence[CandidatePair]) -> list[CandidatePair]:
    return list(dict.fromkeys(pairs))
