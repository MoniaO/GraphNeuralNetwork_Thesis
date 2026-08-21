"""Fusion88 edge decoder (graph branch + optional statistical block).

What it does
------------
Takes node embeddings (from the encoder) and a (source, target) pair.
Builds q_AG = [z_A || z_G || z_A*z_G || |z_A−z_G|] and compresses to g_graph ∈ R^64.
Appends g_stat ∈ R^24 (zeros in Stage A; MLP/KAN in Stage C / FINAL).
Fusion 88 → 64 → 1 logit.

  q_AG (4*d) → Linear → 128 → GELU → Dropout → Linear → 64 → GELU → LayerNorm
  [g_graph || g_stat] ∈ R^88 → 64 → GELU → Dropout → 1 logit

What you may change (new experiment)
------------------------------------
- decoder dropout (default 0.2) — `model.decoder.dropout`
- hidden_channels must match the encoder (FINAL = 32)

What not to touch for FINAL 14.08
---------------------------------
GRAPH_OUT=64, STAT_DIM=24, FUSION_DIM=88, GRAPH_MID=128.
Changing these numbers breaks comparability with the 14.08 table.
"""

from __future__ import annotations

import torch
import torch.nn as nn

STAT_DIM = 24
GRAPH_OUT = 64
FUSION_DIM = GRAPH_OUT + STAT_DIM  # 88
GRAPH_MID = 128


class Fusion88Decoder(nn.Module):
    """Graph pair branch + optional statistical 24-D block."""

    def __init__(self, hidden_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.hidden_channels = int(hidden_channels)
        self.hgt_in = 4 * self.hidden_channels
        self.stat_dim = STAT_DIM
        self.fusion_dim = FUSION_DIM

        self.graph_branch = nn.Sequential(
            nn.Linear(self.hgt_in, GRAPH_MID),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(GRAPH_MID, GRAPH_OUT),
            nn.GELU(),
            nn.LayerNorm(GRAPH_OUT),
        )
        self.output_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        g_stat: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        source = source.to(device=embeddings.device, dtype=torch.long).view(-1)
        target = target.to(device=embeddings.device, dtype=torch.long).view(-1)
        if source.numel() == 0:
            return embeddings.new_empty((0,))

        z_a = embeddings[source]
        z_g = embeddings[target]
        q = torch.cat([z_a, z_g, z_a * z_g, torch.abs(z_a - z_g)], dim=1)
        g_graph = self.graph_branch(q)

        batch = g_graph.size(0)
        if g_stat is None:
            g_stat = g_graph.new_zeros((batch, self.stat_dim))
        else:
            g_stat = g_stat.to(device=g_graph.device, dtype=g_graph.dtype)
            if g_stat.ndim != 2 or g_stat.size(-1) != self.stat_dim:
                raise ValueError(
                    f"g_stat must be [B,{self.stat_dim}], got {tuple(g_stat.shape)}"
                )
            if g_stat.size(0) != batch:
                raise ValueError(
                    f"g_stat batch {g_stat.size(0)} != graph batch {batch}"
                )

        fused = torch.cat([g_graph, g_stat], dim=1)
        if fused.size(-1) != FUSION_DIM:
            raise RuntimeError(f"fusion dim {fused.size(-1)} != {FUSION_DIM}")
        return self.output_head(fused).squeeze(1)


def build_fusion88_decoder(cfg, hidden_channels: int) -> Fusion88Decoder:
    decoder_cfg = getattr(cfg.model, "decoder", None)
    dropout = float(
        getattr(decoder_cfg, "dropout", 0.2) if decoder_cfg is not None else 0.2
    )
    return Fusion88Decoder(hidden_channels=int(hidden_channels), dropout=dropout)
