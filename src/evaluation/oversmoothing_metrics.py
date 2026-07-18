
"""Oversmoothing diagnostics for heterogeneous GNN node embeddings.

Standalone module, independent of any specific evaluator or training loop.
All functions accept raw tensors / z_dict + HeteroData and return plain
Python floats, so they can be reused in notebooks, unit tests, or other
evaluation pipelines without instantiating SynEvaluator.

Metrics:
- mean_pairwise_cosine_similarity: directional collapse (values -> 1.0
  indicate embeddings point in the same direction).
- mean_average_distance (MAD): magnitude-aware pairwise distance, normalized
  by embedding norm.
- numerical_rank: effective rank via SVD energy threshold. Values -> 0
  indicate embeddings collapse into a low-dimensional subspace.
- dirichlet_energy: mean ||z_src - z_dst||^2 over graph edges, normalized by
  embedding dimension. Low values indicate connected nodes become
  indistinguishable (Rusch et al., 2023).
- feature_std: mean per-dimension standard deviation across nodes.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


def mean_pairwise_cosine_similarity(z: torch.Tensor) -> float:
    z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)
    n = z_norm.size(0)
    sum_vec = z_norm.sum(dim=0)
    sum_sq_norm = float((sum_vec @ sum_vec).item())
    total_pairs = n * (n - 1)
    if total_pairs == 0:
        return float("nan")
    return float((sum_sq_norm - n) / total_pairs)


def mean_average_distance(z: torch.Tensor) -> float:
    n = z.size(0)
    if n < 2:
        return float("nan")
    sq_norms = (z ** 2).sum(dim=-1)
    gram = z @ z.t()
    dist_sq = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * gram
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist = torch.sqrt(dist_sq + 1e-12)
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    mean_dist = dist[mask].mean().item()
    mean_norm = z.norm(dim=-1).mean().item() + 1e-8
    return float(mean_dist / mean_norm)


def numerical_rank(z: torch.Tensor, energy_threshold: float = 0.99) -> float:
    z_centered = z - z.mean(dim=0, keepdim=True)
    try:
        _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
    except RuntimeError:
        return float("nan")
    energy = s ** 2
    total_energy = energy.sum().item()
    if total_energy <= 1e-12:
        return 0.0
    cumulative = torch.cumsum(energy, dim=0) / total_energy
    rank = int(torch.searchsorted(cumulative, energy_threshold).item()) + 1
    max_rank = min(z.size(0), z.size(1))
    return float(rank / max_rank) if max_rank > 0 else float("nan")


def dirichlet_energy(z_dict: Dict[str, torch.Tensor], data: HeteroData) -> Optional[float]:
    energies = []
    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        if src_type not in z_dict or dst_type not in z_dict:
            continue
        src_idx, dst_idx = edge_index
        z_src = z_dict[src_type][src_idx]
        z_dst = z_dict[dst_type][dst_idx]
        diff_sq = ((z_src - z_dst) ** 2).sum(dim=-1)
        dim = z_src.size(-1)
        energies.append(float(diff_sq.mean().item() / max(dim, 1)))
    if not energies:
        return None
    return float(np.mean(energies))


def compute_oversmoothing_metrics(
    z_dict: Dict[str, torch.Tensor],
    data: HeteroData,
    sample_size: int = 2000,
) -> Dict[str, float]:
    """Compute all oversmoothing metrics per node type, plus mean across types
    and graph-wide Dirichlet energy.

    sample_size caps the number of nodes used for the O(N^2) pairwise
    computations (cosine similarity, MAD) to keep this tractable on large
    graphs.
    """
    out: Dict[str, float] = {}
    per_type_cosine, per_type_mad, per_type_rank, per_type_std = [], [], [], []

    for node_type, z in z_dict.items():
        if z.numel() == 0 or z.size(0) < 2:
            continue

        n = z.size(0)
        if n > sample_size:
            idx = torch.randperm(n, device=z.device)[:sample_size]
            z_sample = z[idx]
        else:
            z_sample = z

        cosine_sim = mean_pairwise_cosine_similarity(z_sample)
        mad = mean_average_distance(z_sample)
        rank = numerical_rank(z_sample)
        std_mean = float(z_sample.std(dim=0).mean().item())

        out[f"oversmoothing/cosine_sim_{node_type}"] = cosine_sim
        out[f"oversmoothing/mad_{node_type}"] = mad
        out[f"oversmoothing/numerical_rank_{node_type}"] = rank
        out[f"oversmoothing/feature_std_{node_type}"] = std_mean

        per_type_cosine.append(cosine_sim)
        per_type_mad.append(mad)
        per_type_rank.append(rank)
        per_type_std.append(std_mean)

    if per_type_cosine:
        out["oversmoothing/cosine_sim_mean"] = float(np.mean(per_type_cosine))
        out["oversmoothing/mad_mean"] = float(np.mean(per_type_mad))
        out["oversmoothing/numerical_rank_mean"] = float(np.mean(per_type_rank))
        out["oversmoothing/feature_std_mean"] = float(np.mean(per_type_std))

    energy = dirichlet_energy(z_dict, data)
    if energy is not None:
        out["oversmoothing/dirichlet_energy"] = energy

    return out
