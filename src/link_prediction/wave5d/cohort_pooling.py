"""Log-mean-exp cohort aggregation over patient logits."""

from __future__ import annotations

import torch


def log_mean_exp_pool(
    patient_logits: torch.Tensor,
    kappa: float,
    dim: int = 0,
) -> torch.Tensor:
    """
    patient_logits: [n_patients, n_edges]
    kappa: small → mean-like; large → max-like
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive.")

    scaled = kappa * patient_logits
    n = patient_logits.shape[dim]
    pooled = (
        torch.logsumexp(scaled, dim=dim)
        - torch.log(
            torch.tensor(
                float(n),
                device=patient_logits.device,
                dtype=patient_logits.dtype,
            )
        )
    ) / kappa
    return pooled
