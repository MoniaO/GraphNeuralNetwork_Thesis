from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(model: str, hcr: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="config",
            overrides=[f"model={model}", f"hcr={hcr}", "wandb.enabled=false"],
        )


def test_final_aliases_match_historical_source_configs():
    final = _compose("TaskA_hgt_final", "final_ghcr")
    historical = _compose("TaskA_hgt_wave7c", "w7c_b2_audit")

    assert OmegaConf.to_container(final.model, resolve=True) == OmegaConf.to_container(
        historical.model, resolve=True
    )
    assert OmegaConf.to_container(final.hcr, resolve=True) == OmegaConf.to_container(
        historical.hcr, resolve=True
    )
    assert final.model.decoder.arch == "unshared_mlp"
    assert final.model.decoder.pair_encoder.type == "mlp"
    assert final.hcr.encoder == "wave7c_b2_audit"
