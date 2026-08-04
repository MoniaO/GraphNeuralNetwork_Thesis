#!/usr/bin/env python3
"""Wave 5D L2 architecture audit runner (staged).

Stages:
  0 — reproduce baseline L2 on 5 seeds
  1 — one-factor screening (21 × 3 = 63)
  2 — joint encoder confirmation (16 × 5 = 80)  ← CSV + W&B
  3 — final top-3 confirmation (3 × 5 = 15)     ← CSV + W&B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

BASELINE = {
    "model.hgt.hidden_dim": 64,
    "model.hgt.num_layers": 1,
    "model.hgt.heads": 8,
    "model.hgt.activation": "gelu",
    "model.hgt.dropout": 0.2,
    "model.hgt.residual": True,
    "model.decoder.hidden_dims": [64],
    "model.decoder.activation": "gelu",
    "model.decoder.dropout": 0.2,
}

SCREENING_SEEDS = [20260721, 20260722, 20260723]
ALL_SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725]

# From Stage-1 valid AUPRC ranking (frozen for Stage 2).
STAGE2_LAYERS = [1, 3]  # L1 best; L3 second among depths
STAGE2_HEADS = [8, 4]  # baseline 8; heads_4 next
STAGE2_WIDTHS = [32, 64]  # width_32 best; 64 baseline
STAGE2_ENCODER_ACTS = ["leaky_relu", "relu"]  # top two encoder acts
STAGE2_DECODER = {
    "model.decoder.hidden_dims": [256, 128],  # best decoder from Stage 1
    "model.decoder.activation": "gelu",
    "model.decoder.dropout": 0.2,
}


def hydra_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def create_screening_configs():
    configurations = [("baseline", {})]
    for layers in [2, 3]:
        configurations.append((f"depth_{layers}", {"model.hgt.num_layers": layers}))
    for heads in [1, 2, 4]:
        configurations.append((f"heads_{heads}", {"model.hgt.heads": heads}))
    for hidden_dim in [32, 96, 128]:
        configurations.append(
            (f"width_{hidden_dim}", {"model.hgt.hidden_dim": hidden_dim})
        )
    for activation in ["relu", "leaky_relu", "silu", "mish"]:
        configurations.append(
            (f"encoder_act_{activation}", {"model.hgt.activation": activation})
        )
    for activation in ["relu", "leaky_relu", "silu", "mish"]:
        configurations.append(
            (f"decoder_act_{activation}", {"model.decoder.activation": activation})
        )
    for name, hidden_dims in {
        "decoder_128": [128],
        "decoder_256": [256],
        "decoder_128_64": [128, 64],
        "decoder_256_128": [256, 128],
    }.items():
        configurations.append((name, {"model.decoder.hidden_dims": hidden_dims}))
    return configurations


def create_joint_configs():
    """Stage 2: 2×2×2×2 encoder grid; decoder frozen at Stage-1 winner."""
    configs = []
    for layers, heads, width, act in product(
        STAGE2_LAYERS, STAGE2_HEADS, STAGE2_WIDTHS, STAGE2_ENCODER_ACTS
    ):
        tag = f"joint_L{layers}_H{heads}_W{width}_{act}"
        overrides = {
            "model.hgt.num_layers": layers,
            "model.hgt.heads": heads,
            "model.hgt.hidden_dim": width,
            "model.hgt.activation": act,
            **STAGE2_DECODER,
        }
        configs.append((tag, overrides))
    return configs


def write_pipeline_check(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = """WAVE 5D L2 ARCHITECTURE AUDIT
Use: ./run_TaskA_L2_ARCH_AUDIT.sh {check|0|1|2|3|23}
NOT ./run_TaskA_ARCH_DEPTH.sh (Wave2 / no_hcr).
"""
    (out_dir / "PIPELINE_CHECK.txt").write_text(text)


def _result_from_ckpt(tag: str, seed: int, ckpt: Path, overrides: dict) -> dict:
    import torch

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    valid = payload.get("final_valid_metrics") or {}
    test = payload.get("final_test_metrics") or {}
    return {
        "arch_tag": tag,
        "seed": int(seed),
        "best_valid_auprc": float(payload.get("best_valid_metric", float("nan"))),
        "threshold_valid": float(payload.get("classification_threshold", float("nan"))),
        "valid_auprc": float(valid.get("auprc", payload.get("best_valid_metric", float("nan")))),
        "valid_auroc": float(valid.get("auroc", valid.get("auc", float("nan")))),
        "valid_brier": float(valid.get("brier", float("nan"))),
        "test_auprc": float(test.get("auprc", float("nan"))),
        "test_auroc": float(test.get("auroc", test.get("auc", float("nan")))),
        "test_brier": float(test.get("brier", float("nan"))),
        "num_layers": overrides.get("model.hgt.num_layers"),
        "heads": overrides.get("model.hgt.heads"),
        "hidden_dim": overrides.get("model.hgt.hidden_dim"),
        "encoder_activation": overrides.get("model.hgt.activation"),
        "decoder_hidden_dims": json.dumps(
            overrides.get("model.decoder.hidden_dims", BASELINE["model.decoder.hidden_dims"])
        ),
        "decoder_activation": overrides.get(
            "model.decoder.activation", BASELINE["model.decoder.activation"]
        ),
    }


def append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        # replace same arch_tag+seed if re-run
        key = (row["arch_tag"], int(row["seed"]))
        if {"arch_tag", "seed"}.issubset(df_old.columns):
            mask = ~(
                (df_old["arch_tag"].astype(str) == key[0])
                & (df_old["seed"].astype(int) == key[1])
            )
            df_old = df_old.loc[mask]
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)


def run_train(
    tag: str,
    seed: int,
    overrides: dict,
    *,
    epochs: int = 300,
    stage_dir: str = "screen",
    job_type: str = "architecture_audit",
    per_seed_csv: Path | None = None,
) -> Path:
    merged = {**BASELINE, **overrides}
    output_dir = ROOT / "outputs/taskA_arch_audit" / stage_dir / tag / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    if not metrics_path.exists():
        command = [
            PY,
            str(ROOT / "src/train_taskA.py"),
            "model=TaskA_hgt_l2_arch",
            "hcr=structural_latent_pairwise",
            "data.dataset.scenario=clean",
            "data.feature_ablation_profile=empirical",
            "data.candidate_seed=20260722",
            f"training.seed={seed}",
            "training.device=cpu",
            f"training.epochs={epochs}",
            "training.early_stopping_patience=40",
            "experiment.wave=WAVE5D_ARCH_AUDIT",
            f"experiment.intervention={tag}",
            "experiment.motif_completion.enabled=false",
            "wandb.enabled=true",
            "wandb.group=TaskA_WAVE5D_L2_ARCH_AUDIT",
            f"wandb.job_type={job_type}",
            f"hydra.run.dir={output_dir}",
        ]
        command.extend(f"{key}={hydra_value(value)}" for key, value in merged.items())
        if "model.hgt.num_layers" in merged:
            command.append(f"model.num_layers={merged['model.hgt.num_layers']}")
        if "model.hgt.hidden_dim" in merged:
            command.append(f"model.hidden_dim={merged['model.hgt.hidden_dim']}")
            command.append(f"model.hidden_channels={merged['model.hgt.hidden_dim']}")
        if "model.hgt.heads" in merged:
            command.append(f"model.heads={merged['model.hgt.heads']}")

        print("\nRUN:", " ".join(command), flush=True)
        proc = subprocess.run(command, cwd=str(ROOT))
        if proc.returncode != 0:
            raise SystemExit(f"Training failed for {tag} seed={seed} rc={proc.returncode}")

        ckpt = output_dir / "best_model.pt"
        if ckpt.exists():
            result = _result_from_ckpt(tag, seed, ckpt, merged)
            metrics_path.write_text(json.dumps(result, indent=2, default=str))
        else:
            raise SystemExit(f"Missing checkpoint for {tag} seed={seed}")
    else:
        print(f"SKIP {tag} seed={seed} (metrics exist)", flush=True)

    # Always refresh flat CSV row from metrics.json
    row = json.loads(metrics_path.read_text())
    # ensure flat schema
    if "valid_auprc" not in row and "valid" in row:
        row = {
            "arch_tag": tag,
            "seed": seed,
            "best_valid_auprc": row.get("best_valid_auprc"),
            "threshold_valid": row.get("threshold_valid"),
            "valid_auprc": float((row.get("valid") or {}).get("auprc", row.get("best_valid_auprc"))),
            "valid_auroc": float((row.get("valid") or {}).get("auroc", float("nan"))),
            "valid_brier": float((row.get("valid") or {}).get("brier", float("nan"))),
            "test_auprc": float((row.get("test") or {}).get("auprc", float("nan"))),
            "test_auroc": float((row.get("test") or {}).get("auroc", float("nan"))),
            "test_brier": float((row.get("test") or {}).get("brier", float("nan"))),
            "num_layers": merged.get("model.hgt.num_layers"),
            "heads": merged.get("model.hgt.heads"),
            "hidden_dim": merged.get("model.hgt.hidden_dim"),
            "encoder_activation": merged.get("model.hgt.activation"),
            "decoder_hidden_dims": json.dumps(merged.get("model.decoder.hidden_dims")),
            "decoder_activation": merged.get("model.decoder.activation"),
        }
        metrics_path.write_text(json.dumps(row, indent=2, default=str))
    if per_seed_csv is not None:
        append_csv_row(per_seed_csv, row)
    return metrics_path


def summarize_table(per_seed_csv: Path, summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_seed_csv)
    summary = (
        df.groupby("arch_tag", as_index=False)
        .agg(
            n=("seed", "count"),
            valid_auprc_mean=("valid_auprc", "mean"),
            valid_auprc_std=("valid_auprc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_std=("test_auprc", "std"),
            test_brier_mean=("test_brier", "mean"),
            test_brier_std=("test_brier", "std"),
        )
        .sort_values("valid_auprc_mean", ascending=False)
    )
    summary.to_csv(summary_csv, index=False)
    print(summary.to_string(index=False))
    return summary


def upload_wandb_tables(
    *,
    stage: str,
    per_seed_csv: Path,
    summary_csv: Path,
    extra_json: Path | None = None,
) -> str | None:
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skip table upload")
        return None

    run = wandb.init(
        entity="politechnika-gnn-thesis",
        project="politechnika-gnn-thesis",
        group="TaskA_WAVE5D_L2_ARCH_AUDIT",
        job_type=f"architecture_{stage}_summary",
        name=f"l2_arch_{stage}_summary",
        tags=["TaskA", "WAVE5D", "L2_structural_hcr", "arch_audit", stage, "summary"],
        config={"stage": stage, "wave": "WAVE5D_ARCH_AUDIT"},
        reinit=True,
    )
    if per_seed_csv.exists():
        wandb.log({f"{stage}_per_seed": wandb.Table(dataframe=pd.read_csv(per_seed_csv))})
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        wandb.log({f"{stage}_summary": wandb.Table(dataframe=summary)})
        if len(summary):
            top = summary.iloc[0]
            wandb.summary["top_arch_tag"] = str(top["arch_tag"])
            wandb.summary["top_valid_auprc_mean"] = float(top["valid_auprc_mean"])
            wandb.summary["top_test_auprc_mean"] = float(top["test_auprc_mean"])
    if extra_json is not None and extra_json.exists():
        payload = json.loads(extra_json.read_text())
        for k, v in payload.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                wandb.summary[f"decision/{k}"] = v
    url = run.url
    wandb.finish()
    print(f"W&B summary: {url}")
    return url


def summarize_stage0(out_dir: Path, tol: float = 0.01) -> bool:
    rows = []
    for seed in ALL_SEEDS:
        p = out_dir / "screen" / "baseline" / f"seed_{seed}" / "metrics.json"
        if not p.exists():
            print(f"Missing Stage0 metrics: {p}")
            return False
        rows.append(json.loads(p.read_text()))
    df = pd.DataFrame(
        [
            {
                "seed": r.get("seed"),
                "valid_auprc": float(
                    r.get("valid_auprc")
                    or (r.get("valid") or {}).get("auprc", r.get("best_valid_auprc"))
                ),
                "test_auprc": float(
                    r.get("test_auprc")
                    or (r.get("test") or {}).get("auprc", float("nan"))
                ),
                "test_brier": float(
                    r.get("test_brier")
                    or (r.get("test") or {}).get("brier", float("nan"))
                ),
            }
            for r in rows
        ]
    )
    df.to_csv(out_dir / "stage0_reproduce.csv", index=False)
    test_mean = float(df["test_auprc"].mean())
    valid_mean = float(df["valid_auprc"].mean())
    ok = valid_mean >= 0.70 and test_mean >= 0.70
    decision = {
        "stage": 0,
        "test_auprc_mean": test_mean,
        "valid_auprc_mean": valid_mean,
        "reproduce_ok": ok,
        "note": "OK" if ok else "STOP",
    }
    (out_dir / "STAGE0_DECISION.json").write_text(json.dumps(decision, indent=2))
    upload_wandb_tables(
        stage="stage0",
        per_seed_csv=out_dir / "stage0_reproduce.csv",
        summary_csv=out_dir / "stage0_reproduce.csv",
        extra_json=out_dir / "STAGE0_DECISION.json",
    )
    return ok


def summarize_screening(out_dir: Path) -> None:
    rows = []
    for tag, _ in create_screening_configs():
        for seed in SCREENING_SEEDS:
            p = out_dir / "screen" / tag / f"seed_{seed}" / "metrics.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            rows.append(
                {
                    "arch_tag": tag,
                    "seed": seed,
                    "valid_auprc": float(
                        r.get("valid_auprc")
                        or (r.get("valid") or {}).get("auprc", r.get("best_valid_auprc"))
                    ),
                    "test_auprc": float(
                        r.get("test_auprc")
                        or (r.get("test") or {}).get("auprc", float("nan"))
                    ),
                    "test_brier": float(
                        r.get("test_brier")
                        or (r.get("test") or {}).get("brier", float("nan"))
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage1_screening_per_seed.csv", index=False)
    summary = (
        df.groupby("arch_tag", as_index=False)
        .agg(
            n=("seed", "count"),
            valid_auprc_mean=("valid_auprc", "mean"),
            valid_auprc_std=("valid_auprc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_brier_mean=("test_brier", "mean"),
        )
        .sort_values("valid_auprc_mean", ascending=False)
    )
    summary.to_csv(out_dir / "stage1_screening_summary.csv", index=False)
    print(summary.to_string(index=False))
    upload_wandb_tables(
        stage="stage1",
        per_seed_csv=out_dir / "stage1_screening_per_seed.csv",
        summary_csv=out_dir / "stage1_screening_summary.csv",
    )


def run_stage2(out_dir: Path, epochs: int) -> pd.DataFrame:
    selection = {
        "layers": STAGE2_LAYERS,
        "heads": STAGE2_HEADS,
        "widths": STAGE2_WIDTHS,
        "encoder_activations": STAGE2_ENCODER_ACTS,
        "decoder_hidden_dims": STAGE2_DECODER["model.decoder.hidden_dims"],
        "decoder_activation": STAGE2_DECODER["model.decoder.activation"],
        "n_configs": 16,
        "n_seeds": 5,
        "n_runs": 80,
        "selection_split": "valid",
    }
    (out_dir / "STAGE2_SELECTION.json").write_text(json.dumps(selection, indent=2))
    print("Stage 2 selection:", json.dumps(selection, indent=2))

    per_seed = out_dir / "stage2_joint_per_seed.csv"
    # Rebuild CSV from completed metrics (skip trained runs via metrics.json).
    if per_seed.exists():
        per_seed.unlink()

    configs = create_joint_configs()
    for tag, overrides in configs:
        for seed in ALL_SEEDS:
            run_train(
                tag,
                seed,
                overrides,
                epochs=epochs,
                stage_dir="joint",
                job_type="architecture_joint",
                per_seed_csv=per_seed,
            )

    summary = summarize_table(per_seed, out_dir / "stage2_joint_summary.csv")
    decision = {
        "stage": 2,
        "top3": summary.head(3)["arch_tag"].tolist(),
        "top_valid_auprc": float(summary.iloc[0]["valid_auprc_mean"]),
        "n_rows": int(len(pd.read_csv(per_seed))),
    }
    (out_dir / "STAGE2_DECISION.json").write_text(json.dumps(decision, indent=2))
    upload_wandb_tables(
        stage="stage2",
        per_seed_csv=per_seed,
        summary_csv=out_dir / "stage2_joint_summary.csv",
        extra_json=out_dir / "STAGE2_DECISION.json",
    )
    return summary


def run_stage3(out_dir: Path, epochs: int) -> None:
    dec_path = out_dir / "STAGE2_DECISION.json"
    if not dec_path.exists():
        raise SystemExit("Stage 2 decision missing — run --stage 2 first")
    top3 = json.loads(dec_path.read_text())["top3"]
    joint_map = {tag: ov for tag, ov in create_joint_configs()}
    per_seed = out_dir / "stage3_final_per_seed.csv"
    if per_seed.exists():
        per_seed.unlink()

    for tag in top3:
        if tag not in joint_map:
            raise SystemExit(f"Unknown joint tag in top3: {tag}")
        for seed in ALL_SEEDS:
            # Re-run under final/ for a clean Stage-3 artifact (or reuse joint if present)
            run_train(
                tag,
                seed,
                joint_map[tag],
                epochs=epochs,
                stage_dir="final",
                job_type="architecture_final",
                per_seed_csv=per_seed,
            )

    summary = summarize_table(per_seed, out_dir / "stage3_final_summary.csv")
    winner = summary.iloc[0]
    # Tie-break: valid mean, then lower std, then simpler (prefer fewer layers / smaller width)
    decision = {
        "stage": 3,
        "winner_arch_tag": str(winner["arch_tag"]),
        "valid_auprc_mean": float(winner["valid_auprc_mean"]),
        "test_auprc_mean": float(winner["test_auprc_mean"]),
        "test_brier_mean": float(winner["test_brier_mean"]),
        "top3": top3,
        "note": "Architecture frozen by valid AUPRC; threshold re-selected per seed on valid only.",
        "result_label": "L2_ARCH_FROZEN",
    }
    (out_dir / "STAGE3_DECISION.json").write_text(json.dumps(decision, indent=2))
    # Flatten winner config
    tag = str(winner["arch_tag"])
    (out_dir / "FROZEN_L2_ARCHITECTURE.json").write_text(
        json.dumps({"arch_tag": tag, "overrides": joint_map[tag], **decision}, indent=2)
    )
    upload_wandb_tables(
        stage="stage3",
        per_seed_csv=per_seed,
        summary_csv=out_dir / "stage3_final_summary.csv",
        extra_json=out_dir / "STAGE3_DECISION.json",
    )
    print(json.dumps(decision, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["check", "0", "1", "2", "3", "23", "all"],
        default="check",
    )
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    out_dir = ROOT / "outputs/taskA_arch_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pipeline_check(out_dir)

    if args.stage == "check":
        print("Stage 2 joint configs:")
        for tag, ov in create_joint_configs():
            print(" ", tag, ov)
        return

    if args.stage in {"0", "all"}:
        for seed in ALL_SEEDS:
            run_train("baseline", seed, {}, epochs=args.epochs, per_seed_csv=None)
        summarize_stage0(out_dir)

    if args.stage in {"1", "all"}:
        for tag, overrides in create_screening_configs():
            for seed in SCREENING_SEEDS:
                run_train(tag, seed, overrides, epochs=args.epochs)
        summarize_screening(out_dir)

    if args.stage in {"2", "23", "all"}:
        run_stage2(out_dir, args.epochs)

    if args.stage in {"3", "23", "all"}:
        run_stage3(out_dir, args.epochs)


if __name__ == "__main__":
    main()
