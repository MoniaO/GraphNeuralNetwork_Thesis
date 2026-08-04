"""Weights & Biases helpers for Wave 5D."""

from __future__ import annotations

from typing import Any

import pandas as pd

ENTITY = "politechnika-gnn-thesis"
PROJECT = "politechnika-gnn-thesis"
DEFAULT_GROUP = "TaskA_WAVE5D_PATH_LINK"


def wandb_cfg(cfg: dict) -> dict:
    w = dict(cfg.get("wandb") or {})
    return {
        "entity": w.get("entity", ENTITY),
        "project": w.get("project", PROJECT),
        "group": w.get("group", DEFAULT_GROUP),
        "job_type": w.get("job_type", "patient_path_link"),
        "enabled": bool(w.get("enabled", True)),
        "tags": list(w.get("tags") or ["TaskA", "WAVE5D", "link_prediction"]),
    }


def init_variant_run(
    cfg: dict,
    *,
    variant: str,
    seed: int,
    extra_config: dict[str, Any] | None = None,
):
    import wandb

    w = wandb_cfg(cfg)
    if not w["enabled"]:
        return None
    config = {
        "wave": "WAVE5D",
        "variant": variant,
        "training_seed": int(seed),
        "scenario": (cfg.get("data") or {}).get("scenario", "clean"),
        "decoder": cfg.get("decoder"),
        "path_support": cfg.get("path_support"),
        "cohort_pooling": cfg.get("cohort_pooling"),
    }
    if extra_config:
        config.update(extra_config)
    run = wandb.init(
        entity=w["entity"],
        project=w["project"],
        group=w["group"],
        job_type=w["job_type"],
        name=f"wave5d_{variant}_seed{seed}",
        tags=w["tags"] + [variant],
        config=config,
        reinit=True,
    )
    return run


def log_epoch(history_row: dict[str, Any], step: int) -> None:
    import wandb

    if wandb.run is None:
        return
    wandb.log(
        {
            "train/loss": history_row.get("train_loss"),
            "valid/auprc_raw": history_row.get("valid_auprc"),
        },
        step=step,
    )


def log_split_metrics(split: str, metrics: dict[str, Any], role_df: pd.DataFrame | None = None) -> None:
    import wandb

    if wandb.run is None:
        return
    payload = {}
    for key in (
        "auprc",
        "auroc",
        "brier",
        "log_loss",
        "mrr",
        "hits_at_1",
        "hits_at_3",
        "hits_at_5",
        "hits_at_10",
        "n",
        "n_pos",
        "prevalence",
    ):
        if key in metrics and metrics[key] is not None:
            payload[f"{split}/{key}"] = metrics[key]
    if payload:
        wandb.log(payload)
        for k, v in payload.items():
            wandb.run.summary[k] = v
    if role_df is not None and len(role_df):
        for _, row in role_df.iterrows():
            role = str(row["negative_role"])
            if pd.notna(row.get("fpr")):
                wandb.run.summary[f"{split}/fpr/{role}"] = float(row["fpr"])


def finish() -> None:
    import wandb

    if wandb.run is not None:
        wandb.finish()


def log_summary_tables(cfg: dict, summary_dir) -> str | None:
    """Upload aggregate Wave 5D tables as one summary run. Returns run URL."""
    import wandb

    w = wandb_cfg(cfg)
    if not w["enabled"]:
        return None
    run = wandb.init(
        entity=w["entity"],
        project=w["project"],
        group=w["group"],
        job_type="summary",
        name="wave5d_clean_summary",
        tags=w["tags"] + ["summary"],
        config={
            "wave": "WAVE5D",
            "scenario": (cfg.get("data") or {}).get("scenario", "clean"),
            "summary": True,
        },
        reinit=True,
    )
    summary_dir = Path_safe(summary_dir)
    for name in (
        "variant_summary.csv",
        "per_seed_metrics.csv",
        "paired_deltas.csv",
        "role_specific_fpr.csv",
    ):
        path = summary_dir / name
        if path.exists():
            df = pd.read_csv(path)
            wandb.log({name.replace(".csv", ""): wandb.Table(dataframe=df)})
    decision = summary_dir / "TASK_A_LINK_PREDICTION_DECISION.json"
    if decision.exists():
        import json

        dec = json.loads(decision.read_text())
        for k, v in dec.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                wandb.run.summary[f"decision/{k}"] = v
        if isinstance(dec.get("checks"), dict):
            for k, v in dec["checks"].items():
                wandb.run.summary[f"decision/checks/{k}"] = v
    url = run.url
    wandb.finish()
    return url


def Path_safe(p):
    from pathlib import Path

    return Path(p)
