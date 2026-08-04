"""Edge / path potentials for WNERW (nested HGT→HCR log-weights)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def safe_log_probability(
    probability: float,
    epsilon: float = 1e-6,
) -> float:
    probability = min(max(float(probability), epsilon), 1.0 - epsilon)
    return math.log(probability)


def safe_logit(probability: float, epsilon: float = 1e-6) -> float:
    probability = min(max(float(probability), epsilon), 1.0 - epsilon)
    return math.log(probability / (1.0 - probability))


def nested_edge_log_weight(
    p_hgt: float,
    p_hcr2: float | None,
    p_hcr3: float | None,
    uncertainty: float,
    *,
    alpha: float = 1.0,
    beta_hcr2: float = 1.0,
    beta_hcr3: float = 0.0,
    uncertainty_penalty: float = 0.0,
    length_penalty: float = 0.0,
) -> float:
    """Nested log-evidence for one directed edge.

    With ``alpha = beta_hcr2 = 1`` and ``beta_hcr3 = 0`` this reduces to
    ``log p_hcr2`` (no double-counting of the HGT term).
    """
    log_hgt = safe_log_probability(p_hgt)
    result = alpha * log_hgt

    if p_hcr2 is not None:
        log_hcr2 = safe_log_probability(p_hcr2)
        result += beta_hcr2 * (log_hcr2 - log_hgt)
    else:
        log_hcr2 = log_hgt

    if p_hcr3 is not None:
        log_hcr3 = safe_log_probability(p_hcr3)
        result += beta_hcr3 * (log_hcr3 - log_hcr2)

    result -= uncertainty_penalty * float(uncertainty)
    result -= length_penalty
    return float(result)


def edge_probability_for_graph(
    *,
    edge_in_train: bool,
    p_calibrated: float,
    known_edge_probability: float = 1.0 - 1e-6,
) -> float:
    """Stage-2 default: train edges are nearly certain; predicted use model p."""
    if edge_in_train:
        return float(known_edge_probability)
    return float(p_calibrated)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def build_edge_log_weights(
    edges: list[Mapping[str, Any]],
    *,
    alpha_hgt: float = 1.0,
    beta_hcr2: float = 1.0,
    beta_hcr3: float = 0.0,
    gamma_delta: float = 1.0,
    uncertainty_penalty: float = 0.0,
    length_penalty: float = 0.0,
    known_edge_probability: float = 1.0 - 1e-6,
    probability_key: str = "p_calibrated",
    mode: str = "nested_probabilities",
) -> dict[tuple[str, str], float]:
    """Map ``(source, target) → q_uv`` from an evidence table / edge records.

    Modes
    -----
    uniform
        Constant (length penalty only).
    log_calibrated
        ``q = log p`` from ``probability_key``.
    nested_probabilities
        Nested HGT → HCR2 → HCR3 increments.
    d1_log_prob  (Wave 5B)
        ``q = log P_D1(A→G)`` — decoder probability from structural latent pairwise.
    d1_delta_logit  (Wave 5B)
        ``q = log p_HGT + γ (logit P_D1 − logit p_HGT)``.
    d1_nested  (Wave 5B)
        Nested with ``p_D1`` in place of HCR2: ``α log p_HGT + β (log P_D1 − log p_HGT)``.
    """
    out: dict[tuple[str, str], float] = {}
    for row in edges:
        source = str(row["source"])
        target = str(row["target"])
        key = (source, target)
        in_train = bool(row.get("edge_in_train", False))

        if mode == "uniform":
            out[key] = -float(length_penalty)
            continue

        if in_train:
            p = known_edge_probability
            out[key] = safe_log_probability(p) - float(length_penalty)
            continue

        if mode == "log_calibrated":
            p = float(row[probability_key])
            out[key] = safe_log_probability(p) - float(length_penalty)
            continue

        p_hgt = float(row.get("p_hgt_calibrated", row.get("p_hgt", 0.5)))
        p_d1 = _finite_or_none(row.get("p_d1_calibrated", row.get("p_d1")))
        uncertainty = float(row.get("hcr_uncertainty", 0.0) or 0.0)

        if mode == "d1_log_prob":
            if p_d1 is None:
                out[key] = safe_log_probability(p_hgt) - float(length_penalty)
            else:
                out[key] = (
                    safe_log_probability(p_d1)
                    - float(uncertainty_penalty) * uncertainty
                    - float(length_penalty)
                )
            continue

        if mode == "d1_delta_logit":
            if p_d1 is None:
                out[key] = safe_log_probability(p_hgt) - float(length_penalty)
            else:
                delta = safe_logit(p_d1) - safe_logit(p_hgt)
                out[key] = (
                    safe_log_probability(p_hgt)
                    + float(gamma_delta) * delta
                    - float(uncertainty_penalty) * uncertainty
                    - float(length_penalty)
                )
            continue

        if mode == "d1_nested":
            out[key] = nested_edge_log_weight(
                p_hgt,
                p_d1,
                None,
                uncertainty,
                alpha=alpha_hgt,
                beta_hcr2=beta_hcr2,
                beta_hcr3=0.0,
                uncertainty_penalty=uncertainty_penalty,
                length_penalty=length_penalty,
            )
            continue

        if mode != "nested_probabilities":
            raise ValueError(f"Unknown edge potential mode: {mode!r}")

        p_hcr2 = _finite_or_none(row.get("p_hcr2_calibrated", row.get("p_hcr2")))
        p_hcr3 = _finite_or_none(row.get("p_hcr3_calibrated", row.get("p_hcr3")))

        out[key] = nested_edge_log_weight(
            p_hgt,
            p_hcr2,
            p_hcr3,
            uncertainty,
            alpha=alpha_hgt,
            beta_hcr2=beta_hcr2,
            beta_hcr3=beta_hcr3,
            uncertainty_penalty=uncertainty_penalty,
            length_penalty=length_penalty,
        )
    return out
