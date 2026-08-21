"""Stage A hyperparameter grids for Final Large Grid 11.08.2026."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterator


CANDIDATE_SEED = 20260722
SCREENING_SEEDS = (20260721, 20260722, 20260723)
FINAL_SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)

SHARED_HIDDEN = (32, 64)
SHARED_LAYERS = (2, 3, 4)
SHARED_DROPOUT = (0.10, 0.25)
SHARED_LR = (1e-3, 3e-4)

FIXED = {
    "weight_decay": 5e-4,
    "epochs": 200,
    "patience": 40,
    "grad_clip": 1.0,
}

# Defaults used during shared grid (before architecture-specific refinement).
DEFAULT_HGT_HEADS = 4
DEFAULT_GAT_HEADS = 4
DEFAULT_RGCN_BASES: int | str = 8

REFINE_HGT_HEADS = (4, 8)
REFINE_GAT_HEADS = (2, 4, 8)
REFINE_RGCN_BASES: tuple[int | str, ...] = (4, 8, "full")

BACKBONES = ("rgcn_matched", "hgt", "hetero_gatv2", "hetero_sage_matched")


@dataclass(frozen=True)
class StageAConfig:
    backbone: str
    hidden_dim: int
    n_layers: int
    dropout: float
    lr: float
    heads: int | None = None
    num_bases: int | str | None = None
    phase: str = "shared"  # shared | refine

    @property
    def config_id(self) -> str:
        parts = [
            self.backbone,
            f"h{self.hidden_dim}",
            f"L{self.n_layers}",
            f"d{self.dropout:g}",
            f"lr{self.lr:g}",
        ]
        if self.heads is not None:
            parts.append(f"hd{self.heads}")
        if self.num_bases is not None:
            parts.append(f"b{self.num_bases}")
        parts.append(self.phase)
        return "__".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_shared_grid() -> Iterator[StageAConfig]:
    for backbone, h, L, drop, lr in product(
        BACKBONES, SHARED_HIDDEN, SHARED_LAYERS, SHARED_DROPOUT, SHARED_LR
    ):
        heads = None
        bases: int | str | None = None
        if backbone == "hgt":
            heads = DEFAULT_HGT_HEADS
        elif backbone == "hetero_gatv2":
            heads = DEFAULT_GAT_HEADS
            if h % heads != 0:
                continue
        elif backbone == "rgcn_matched":
            bases = DEFAULT_RGCN_BASES
        yield StageAConfig(
            backbone=backbone,
            hidden_dim=h,
            n_layers=L,
            dropout=drop,
            lr=lr,
            heads=heads,
            num_bases=bases,
            phase="shared",
        )


def iter_refine_grid(backbone: str, base: StageAConfig) -> Iterator[StageAConfig]:
    """Architecture-specific refinement around a frozen shared Top config."""
    if backbone == "hgt":
        for heads in REFINE_HGT_HEADS:
            yield StageAConfig(
                backbone=backbone,
                hidden_dim=base.hidden_dim,
                n_layers=base.n_layers,
                dropout=base.dropout,
                lr=base.lr,
                heads=heads,
                num_bases=None,
                phase="refine",
            )
    elif backbone == "hetero_gatv2":
        for heads in REFINE_GAT_HEADS:
            if base.hidden_dim % heads != 0:
                continue
            yield StageAConfig(
                backbone=backbone,
                hidden_dim=base.hidden_dim,
                n_layers=base.n_layers,
                dropout=base.dropout,
                lr=base.lr,
                heads=heads,
                num_bases=None,
                phase="refine",
            )
    elif backbone == "rgcn_matched":
        for bases in REFINE_RGCN_BASES:
            yield StageAConfig(
                backbone=backbone,
                hidden_dim=base.hidden_dim,
                n_layers=base.n_layers,
                dropout=base.dropout,
                lr=base.lr,
                heads=None,
                num_bases=bases,
                phase="refine",
            )
    else:
        raise ValueError(backbone)


def count_shared_jobs(n_seeds: int = 3) -> int:
    return sum(1 for _ in iter_shared_grid()) * n_seeds
