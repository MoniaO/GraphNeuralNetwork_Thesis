"""Wave 4C latent-gate: no gate-column leakage in HCR triples."""

from __future__ import annotations

from hydra import compose, initialize_config_dir
from pathlib import Path

from experiments.motif_completion import latent_triple_map
from hcr.motif_registry_v3 import GATE_OUTCOMES


ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                "model=TaskA_hgt_hcr",
                "hcr=hcr3_full",
                "hcr.z_mode=oracle_outcome",
                "experiment.motif_completion.enabled=true",
                "experiment.latent_gate.enabled=true",
                "wandb.enabled=false",
            ],
        )


def test_latent_triples_never_use_gate_as_y_or_z():
    cfg = _cfg()
    triples = latent_triple_map(cfg)
    assert len(triples) == 10
    for (a, g), (x, y, z) in triples.items():
        assert x == a
        assert g not in {x, y, z}, f"gate {g} leaked into triple {(x,y,z)}"
        assert y == GATE_OUTCOMES[g]
        assert z  # co-parent present


def test_gate_outcomes_cover_binary_duals():
    from hcr.motif_registry_v3 import DUAL_GATES

    for gate in DUAL_GATES:
        assert gate in GATE_OUTCOMES
