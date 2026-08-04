"""Checkpoint key remapping for Wave 7C → Wave 9/10 MLPPairEncoder."""

from __future__ import annotations

import re
from typing import Mapping

import torch


_ROLE_SEQ = re.compile(r"^(decoder\.role_encoders\.\d+)\.(\d+\..*)$")


def remap_wave7c_mlp_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map legacy ``role_encoders.i.j.*`` → ``role_encoders.i.net.j.*``.

    Pre-Wave-9 A1 checkpoints stored unshared encoders as bare ``nn.Sequential``.
    Current code wraps the same stack in ``MLPPairEncoder.net``.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        m = _ROLE_SEQ.match(k)
        if m and ".net." not in k:
            out[f"{m.group(1)}.net.{m.group(2)}"] = v
        else:
            out[k] = v
    return out
