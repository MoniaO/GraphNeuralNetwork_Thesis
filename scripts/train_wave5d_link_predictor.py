#!/usr/bin/env python3
"""Train one Wave 5D variant (with κ selection on validation) + W&B logging."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from link_prediction.wave5d.final_edge_decoder import VARIANT_MASKS
from link_prediction.wave5d.patient_edge_features import load_feature_bundle
from link_prediction.wave5d.train_loop import TrainConfig, evaluate_split, train_variant
from link_prediction.wave5d.wandb_logging import (
    finish as wandb_finish,
    init_variant_run,
    log_epoch,
    log_split_metrics,
    wandb_cfg,
)


CONTROL_FEATURE_TAG = {
    "L0_hgt": "base",
    "L1_pairwise_hcr": "base",
    "L2_structural_hcr": "base",
    "L3_path_support": "base",
    "L4_final": "base",
    "L5_patient_shuffle": "patient_shuffle",
    "L6_context_shuffle": "context_shuffle",
    "L7_matched_random_context": "matched_random_context",
    "L8_random_path_weights": "random_path_weights",
    "L9_all_context_warning": "base",
    "L10_no_leave_one_out": "no_leave_one_out",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/link_prediction/wave5d_clean.yaml",
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    variant = args.variant
    if variant not in VARIANT_MASKS:
        raise SystemExit(f"Unknown variant {variant}")

    tag = CONTROL_FEATURE_TAG[variant]
    feat_path = (
        ROOT
        / cfg["paths"]["output_dir"]
        / "patient_features"
        / f"features_seed{args.seed}_{tag}.npz"
    )
    if not feat_path.exists():
        raise SystemExit(f"Missing features: {feat_path}")

    bundle = load_feature_bundle(feat_path)
    mask = VARIANT_MASKS[variant]
    dec = cfg["decoder"]
    if args.kappa is not None:
        kappa_grid = [float(args.kappa)]
    elif variant == "L4_final":
        kappa_grid = [float(k) for k in cfg["cohort_pooling"]["kappa_grid"]]
    else:
        kappa_grid = [float(cfg["cohort_pooling"].get("default_kappa", 1.0))]

    os.environ.setdefault("WANDB_MODE", "online")
    run = init_variant_run(
        cfg,
        variant=variant,
        seed=int(args.seed),
        extra_config={"feature_tag": tag, "kappa_grid": kappa_grid},
    )
    if run is not None:
        print(f"W&B run: {run.url}", flush=True)

    best = None
    for kappa in kappa_grid:
        tcfg = TrainConfig(
            hidden_dim=int(dec["hidden_dim"]),
            dropout=float(dec["dropout"]),
            lr=float(dec["lr"]),
            weight_decay=float(dec["weight_decay"]),
            epochs=int(dec["epochs"]),
            batch_edges=int(dec["batch_edges"]),
            kappa=float(kappa),
            seed=int(args.seed),
        )
        model, calibrator, info = train_variant(
            bundle,
            mask,
            tcfg,
            log_epoch_fn=log_epoch if run is not None else None,
        )
        device = torch.device("cpu")
        valid = evaluate_split(
            model, calibrator, bundle, mask, kappa, "valid", device
        )
        score = float(valid["auprc"]) if np.isfinite(valid["auprc"]) else -1.0
        print(f"  kappa={kappa} valid_auprc={score:.4f}", flush=True)
        if run is not None:
            import wandb

            wandb.log({f"kappa_select/valid_auprc@{kappa}": score})
        if best is None or score > best["valid_auprc"]:
            best = {
                "kappa": kappa,
                "model": model,
                "calibrator": calibrator,
                "valid_auprc": score,
                "info": info,
            }

    assert best is not None
    kappa = best["kappa"]
    model = best["model"]
    calibrator = best["calibrator"]
    device = torch.device("cpu")

    if run is not None:
        import wandb

        wandb.run.summary["selected_kappa"] = kappa
        wandb.run.summary["best_valid_auprc"] = best["valid_auprc"]
        wandb.config.update({"selected_kappa": kappa}, allow_val_change=True)

    ckpt_dir = ROOT / cfg["paths"]["output_dir"] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{variant}_seed{args.seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "kappa": kappa,
            "calibrator": {
                "temperature": calibrator.temperature,
                "intercept": calibrator.intercept,
            },
            "variant": variant,
            "seed": args.seed,
            "in_dim": int(model.net[0].in_features),
        },
        ckpt_path,
    )

    pred_dir = ROOT / cfg["paths"]["output_dir"] / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = ROOT / cfg["paths"]["output_dir"] / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for split in ("valid", "test"):
        ev = evaluate_split(model, calibrator, bundle, mask, kappa, split, device)
        role_df = ev.pop("role_fpr")
        role_df.to_csv(
            pred_dir / f"{variant}_seed{args.seed}_{split}_role_fpr.csv",
            index=False,
        )
        idx = ev.pop("index")
        logits = ev.pop("logits")
        probs = ev.pop("probs")
        labels = ev.pop("labels")
        frame = pd.DataFrame(
            {
                "candidate_id": [bundle.candidate_ids[i] for i in idx],
                "source": [bundle.sources[i] for i in idx],
                "target": [bundle.targets[i] for i in idx],
                "edge_type": [bundle.edge_types[i] for i in idx],
                "negative_role": [bundle.negative_roles[i] for i in idx],
                "label": labels,
                "logit": logits,
                "probability": probs,
                "split": split,
                "variant": variant,
                "seed": args.seed,
            }
        )
        frame.to_csv(pred_dir / f"{variant}_seed{args.seed}_{split}.csv", index=False)
        metrics_rows.append(
            {
                "variant": variant,
                "seed": args.seed,
                "split": split,
                "kappa": kappa,
                **{
                    k: v
                    for k, v in ev.items()
                    if not isinstance(v, (pd.DataFrame, np.ndarray))
                },
            }
        )
        log_split_metrics(split, ev, role_df)
        print(
            f"{variant} seed={args.seed} {split}: "
            f"AUPRC={ev['auprc']:.4f} AUROC={ev['auroc']:.4f} Brier={ev['brier']:.4f}",
            flush=True,
        )

    pd.DataFrame(metrics_rows).to_csv(
        pred_dir / f"{variant}_seed{args.seed}_metrics.csv", index=False
    )
    (cal_dir / f"{variant}_seed{args.seed}.json").write_text(
        json.dumps(
            {
                "temperature": calibrator.temperature,
                "intercept": calibrator.intercept,
                "kappa": kappa,
                "valid_auprc": best["valid_auprc"],
                "wandb_enabled": wandb_cfg(cfg)["enabled"],
            },
            indent=2,
        )
    )
    wandb_finish()


if __name__ == "__main__":
    main()
