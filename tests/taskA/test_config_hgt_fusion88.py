from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_final_model_matches_frozen_14_08_stack():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=["model=hgt_fusion88", "hcr=none", "wandb.enabled=false"],
        )

    assert cfg.model.encoder_name == "hgt"
    assert int(cfg.model.hidden_dim) == 32
    assert int(cfg.model.num_layers) == 2
    assert int(cfg.model.heads) == 4
    assert float(cfg.model.dropout) == 0.25
    assert cfg.model.decoder.name == "fusion88_stat"
    assert int(cfg.model.decoder.stat_raw_dim) == 40
    assert cfg.model.decoder.stat_pair_encoder == "mlp"
    assert cfg.hcr.enabled is False
    assert cfg.hcr.variant == "none"
