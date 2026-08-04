from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------
# Ogolne metryki embeddingow
# ---------------------------------------------------------------------

@torch.no_grad()
def mean_pairwise_cosine_similarity(z: torch.Tensor) -> float:
    """Srednie podobienstwo kosinusowe po wszystkich parach wezlow."""
    if z.ndim != 2:
        raise ValueError(f"Oczekiwano [num_nodes, embedding_dim], otrzymano {tuple(z.shape)}.")

    n = int(z.size(0))
    if n < 2 or not torch.isfinite(z).all():
        return float("nan")

    z_norm = F.normalize(z.detach().float(), p=2, dim=-1, eps=1e-8)

    # ||sum_i z_i||^2 = sum_i ||z_i||^2 + sum_{i!=j} z_i . z_j
    summed = z_norm.sum(dim=0)
    total_sq = float((summed @ summed).item())
    diag_sq = float(z_norm.pow(2).sum().item())  # ~n dla znormalizowanych wektorow
    off_diag = total_sq - diag_sq
    n_pairs = n * (n - 1)
    return float(off_diag / n_pairs)


@torch.no_grad()
def mean_average_distance(z: torch.Tensor) -> float:
    """Znormalizowana srednia odleglosc euklidesowa miedzy parami wezlow."""
    if z.ndim != 2:
        raise ValueError(f"Oczekiwano [num_nodes, embedding_dim], otrzymano {tuple(z.shape)}.")

    n = int(z.size(0))
    if n < 2 or not torch.isfinite(z).all():
        return float("nan")

    z_float = z.detach().float()
    sq_norms = z_float.pow(2).sum(dim=-1)
    gram = z_float @ z_float.t()
    sq_dist = torch.clamp(sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * gram, min=0.0)
    dist = torch.sqrt(sq_dist)

    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    mean_dist = float(dist[mask].mean().item())
    mean_norm = float(z_float.norm(dim=-1).mean().item())

    if mean_norm <= 1e-12:
        return 0.0
    return float(mean_dist / mean_norm)


@torch.no_grad()
def numerical_rank(z: torch.Tensor, energy_threshold: float = 0.99) -> float:
    """Znormalizowany efektywny rank przez prog energii wartosci osobliwych."""
    if z.ndim != 2:
        raise ValueError(f"Oczekiwano [num_nodes, embedding_dim], otrzymano {tuple(z.shape)}.")
    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError(f"energy_threshold musi byc w (0, 1], otrzymano {energy_threshold}.")

    n, dim = int(z.size(0)), int(z.size(1))
    if n < 2 or dim == 0 or not torch.isfinite(z).all():
        return float("nan")

    z_centered = z.detach().float() - z.detach().float().mean(dim=0, keepdim=True)
    try:
        singular_values = torch.linalg.svdvals(z_centered)
    except RuntimeError:
        return float("nan")

    energy = singular_values.pow(2)
    total_energy = float(energy.sum().item())
    if total_energy <= 1e-12:
        return 0.0

    cumulative = torch.cumsum(energy, dim=0) / total_energy
    threshold_tensor = torch.tensor(energy_threshold, dtype=cumulative.dtype, device=cumulative.device)
    rank = int(torch.searchsorted(cumulative, threshold_tensor).item()) + 1

    # Centrowanie ogranicza maksymalny rank do co najwyzej n-1 (wektory
    # scentrowane sumuja sie do zera - jedno ograniczenie liniowe).
    max_rank = min(n - 1, dim)
    if max_rank <= 0:
        return float("nan")

    rank = min(rank, max_rank)
    return float(rank / max_rank)


# ---------------------------------------------------------------------
# Energia Dirichleta
# ---------------------------------------------------------------------

def _edge_type_name(edge_type: Tuple[str, str, str]) -> str:
    src, rel, dst = edge_type
    return f"{src}__{rel}__{dst}"


@torch.no_grad()
def compute_dirichlet_energy(
    z_dict: Dict[str, torch.Tensor], data: HeteroData
) -> Dict[str, float]:
    """Energia Dirichleta per relacja + dwa warianty agregacji globalnej.

    Dla krawedzi u -> v: energy(u, v) = ||z_u - z_v||^2 / embedding_dim.

    Zwraca:
    - "edge_weighted": srednia po WSZYSTKICH pojedynczych krawedziach
      (relacje z wieksza liczba krawedzi wnosza proporcjonalnie wiecej).
    - "relation_macro": srednia po srednich per-relacyjnych (kazdy typ
      relacji rowny, niezaleznie od liczby krawedzi).
    - "relation/<edge_type>": energia w obrebie jednej relacji heterogenicznej.
    """
    if not isinstance(data, HeteroData):
        raise TypeError("data musi byc instancja torch_geometric.data.HeteroData.")

    output: Dict[str, float] = {}
    relation_energies: list = []
    total_edge_energy, total_edge_count = 0.0, 0

    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        if src_type not in z_dict or dst_type not in z_dict:
            continue

        z_src_all, z_dst_all = z_dict[src_type], z_dict[dst_type]
        if z_src_all.size(-1) != z_dst_all.size(-1):
            raise ValueError(
                f"Relacja {edge_type}: embeddingi src/dst maja rozne wymiary "
                f"({z_src_all.size(-1)} vs {z_dst_all.size(-1)})."
            )
        if z_src_all.device != z_dst_all.device:
            raise ValueError(f"Relacja {edge_type}: embeddingi src/dst na roznych urzadzeniach.")

        device = z_src_all.device
        src_idx, dst_idx = edge_index[0].to(device), edge_index[1].to(device)
        z_src = z_src_all[src_idx].detach().float()
        z_dst = z_dst_all[dst_idx].detach().float()

        relation_name = _edge_type_name(edge_type)

        if not torch.isfinite(z_src).all() or not torch.isfinite(z_dst).all():
            # Zdegenerowana relacja (NaN/Inf) - wykluczona z agregatow,
            # zeby nie zatruc reszty wyniku przez propagacje NaN.
            output[f"relation/{relation_name}"] = float("nan")
            continue

        dim = max(int(z_src.size(-1)), 1)
        per_edge_energy = (z_src - z_dst).pow(2).sum(dim=-1) / dim

        relation_energy = float(per_edge_energy.mean().item())
        output[f"relation/{relation_name}"] = relation_energy
        relation_energies.append(relation_energy)

        total_edge_energy += float(per_edge_energy.sum().item())
        total_edge_count += int(per_edge_energy.numel())

    if total_edge_count > 0:
        output["edge_weighted"] = float(total_edge_energy / total_edge_count)

    finite_relation_energies = [v for v in relation_energies if np.isfinite(v)]
    if finite_relation_energies:
        output["relation_macro"] = float(np.mean(finite_relation_energies))

    return output


# ---------------------------------------------------------------------
# Probkowanie (deterministyczne, bez ingerencji w globalny RNG)
# ---------------------------------------------------------------------

def _sample_embeddings(z: torch.Tensor, sample_size: int, generator: torch.Generator) -> torch.Tensor:
    """Probkuje wezly bez modyfikowania globalnego stanu RNG PyTorch - patrz
    uzasadnienie w docstringu modulu."""
    n = int(z.size(0))
    if n <= sample_size:
        return z
    idx = torch.randperm(n, generator=generator, device="cpu")[:sample_size]
    return z[idx.to(z.device)]


def _finite_mean(values: list) -> Optional[float]:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


# ---------------------------------------------------------------------
# Pelna diagnostyka
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_oversmoothing_metrics(
    z_dict: Dict[str, torch.Tensor],
    data: HeteroData,
    sample_size: int = 2000,
    sample_seed: int = 42,
    energy_threshold: float = 0.99,
) -> Dict[str, float]:
    """Liczy wszystkie metryki oversmoothingu per typ wezla, ich srednie po
    typach, oraz energie Dirichleta (relation_macro / edge_weighted / per
    relacja).

    sample_size ogranicza liczbe wezlow uzywanych do obliczen O(N^2)
    (cosine similarity, MAD) per typ wezla.
    sample_seed - seed WYLACZNIE dla probkowania diagnostycznego, niezalezny
    od globalnego RNG treningu (patrz _sample_embeddings).
    """
    if sample_size <= 0:
        raise ValueError(f"sample_size musi byc > 0, otrzymano {sample_size}.")

    output: Dict[str, float] = {}
    per_type_cosine, per_type_mad, per_type_rank, per_type_std = [], [], [], []

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(sample_seed))

    # Sortowanie zapewnia deterministyczna kolejnosc przetwarzania typow.
    for node_type in sorted(z_dict):
        z = z_dict[node_type]
        if z.ndim != 2:
            raise ValueError(
                f"Embeddingi dla typu {node_type!r} musza miec [num_nodes, dim], "
                f"otrzymano {tuple(z.shape)}."
            )
        if z.numel() == 0 or z.size(0) < 2:
            continue

        z_sample = _sample_embeddings(z, sample_size, generator)

        cosine_sim = mean_pairwise_cosine_similarity(z_sample)
        mad = mean_average_distance(z_sample)
        rank = numerical_rank(z_sample, energy_threshold=energy_threshold)
        std_mean = (
            float(z_sample.detach().float().std(dim=0, unbiased=False).mean().item())
            if torch.isfinite(z_sample).all() else float("nan")
        )

        output[f"oversmoothing/cosine_sim_{node_type}"] = cosine_sim
        output[f"oversmoothing/mad_{node_type}"] = mad
        output[f"oversmoothing/numerical_rank_{node_type}"] = rank
        output[f"oversmoothing/feature_std_{node_type}"] = std_mean

        per_type_cosine.append(cosine_sim)
        per_type_mad.append(mad)
        per_type_rank.append(rank)
        per_type_std.append(std_mean)

    cosine_mean = _finite_mean(per_type_cosine)
    mad_mean = _finite_mean(per_type_mad)
    rank_mean = _finite_mean(per_type_rank)
    std_mean_agg = _finite_mean(per_type_std)

    if cosine_mean is not None:
        output["oversmoothing/cosine_sim_mean"] = cosine_mean
    if mad_mean is not None:
        output["oversmoothing/mad_mean"] = mad_mean
    if rank_mean is not None:
        output["oversmoothing/numerical_rank_mean"] = rank_mean
    if std_mean_agg is not None:
        output["oversmoothing/feature_std_mean"] = std_mean_agg

    dirichlet = compute_dirichlet_energy(z_dict, data)
    if "relation_macro" in dirichlet:
        output["oversmoothing/dirichlet_energy_relation_macro"] = dirichlet["relation_macro"]
        output["oversmoothing/dirichlet_energy"] = dirichlet["relation_macro"]
    if "edge_weighted" in dirichlet:
        output["oversmoothing/dirichlet_energy_edge_weighted"] = dirichlet["edge_weighted"]
    for key, value in dirichlet.items():
        if key.startswith("relation/"):
            relation_name = key[len("relation/"):]
            output[f"oversmoothing/dirichlet_energy_relation_{relation_name}"] = value

    return output
