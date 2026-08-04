#!/usr/bin/env python3
"""Task A full-retrain screen: TA_MLP vs TA_KAN × 6 scenarios × 3 seeds.

SOURCE OF TRUTH: W7D_T0_A1 / W9_K0_MLP_CONTROL
Only `model.decoder.pair_encoder.type` differs (mlp vs kan).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "wave11_taskA" / "mlp_vs_kan_full"
CONTEXT_REGISTRY = ROOT / "outputs" / "wave5c" / "registry" / "edge_context_registry.csv"
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"

SEEDS = (20260722, 20260723, 20260724)
SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)

MODELS = {
    "TA_MLP": ROOT / "configs" / "taskA" / "mlp_full_retrain.yaml",
    "TA_KAN": ROOT / "configs" / "taskA" / "kan_full_retrain.yaml",
}

def _env() -> dict:
    env = os.environ.copy()
    env.setdefault("GSN_PROJECT_ROOT", str(DEFAULT_GSN))
    env.setdefault(
        "PHARMA_DATA_ROOT",
        str(Path(env["GSN_PROJECT_ROOT"]) / "2 v3. Data" / "dataset_v3"),
    )
    env.setdefault("PYTHONPATH", f"{ROOT / 'src'}:{ROOT}")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _peak_memory_mb() -> float:
    # macOS: ru_maxrss is bytes; Linux: kilobytes
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if rss <= 0:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _metrics_from_ckpt(ckpt: Path) -> dict:
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    fv = blob.get("final_valid_metrics") or {}
    ft = blob.get("final_test_metrics") or {}
    n_params = sum(int(v.numel()) for v in blob["model_state_dict"].values())
    hist = blob.get("history") or []
    best_epoch = blob.get("best_epoch")
    if best_epoch is None and hist:
        best_epoch = max(
            (h for h in hist if "epoch" in h),
            key=lambda h: float(h.get("valid_auprc", h.get("auprc", -1))),
            default={},
        ).get("epoch")
    return {
        "best_valid_auprc": float(blob.get("best_valid_metric", float("nan"))),
        "valid_auprc": float(fv.get("auprc", blob.get("best_valid_metric", float("nan")))),
        "valid_brier": float(fv.get("brier", float("nan"))),
        "valid_auroc": float(fv.get("auc", fv.get("auroc", float("nan")))),
        "test_auprc": float(ft.get("auprc", float("nan"))),
        "test_brier": float(ft.get("brier", float("nan"))),
        "test_auroc": float(ft.get("auc", ft.get("auroc", float("nan")))),
        "parameter_count": n_params,
        "best_epoch": best_epoch if best_epoch is not None else float("nan"),
        "candidate_fingerprint": blob.get("candidate_fingerprint"),
        "candidate_seed": blob.get("candidate_seed"),
    }


def build_train_cmd(
    *,
    model_id: str,
    cfg: dict,
    scenario: str,
    seed: int,
    epochs: int,
    run_dir: Path,
) -> list[str]:
    pe = cfg["decoder"]["pair_encoder"]
    pe_type = str(pe["type"])
    cmd = [
        PY,
        str(ROOT / "src" / "train_taskA.py"),
        f"model={cfg['model']}",
        f"hcr={cfg['hcr']}",
        f"data.dataset.scenario={scenario}",
        "data.feature_ablation_profile=empirical",
        f"data.candidate_seed={seed}",
        f"training.seed={seed}",
        f"training.device={cfg['training'].get('device', 'cpu')}",
        f"training.epochs={epochs}",
        f"training.early_stopping_patience={cfg['training'].get('early_stopping_patience', 40)}",
        "training.grad_clip=1.0",
        "training.init_from_checkpoint=null",
        "experiment.wave=TASKA_MLP_VS_KAN",
        f"experiment.variant={cfg['variant']}",
        f"experiment.intervention={cfg['variant']}",
        "experiment.motif_completion.enabled=false",
        "model.decoder.arch=unshared_mlp",
        "model.decoder.ablation=none",
        "model.decoder.use_triple=false",
        "model.decoder.ag_kan_residual.enabled=false",
        f"model.decoder.pair_encoder.type={pe_type}",
        "model.decoder.pair_encoder.input_dim=40",
        "model.decoder.pair_encoder.hidden_dim=16",
        "model.decoder.pair_encoder.output_dim=8",
        f"model.decoder.pair_encoder.dropout={pe.get('dropout', 0.1)}",
        "wandb.enabled=false",
        f"hydra.run.dir={run_dir}",
        "model.hgt.hidden_dim=32",
        "model.hgt.num_layers=3",
        "model.hgt.heads=8",
        "model.hgt.activation=leaky_relu",
        "model.hgt.dropout=0.2",
        "model.hgt.residual=true",
        "model.num_layers=3",
        "model.hidden_dim=32",
        "model.hidden_channels=32",
        "model.heads=8",
    ]
    if pe_type == "kan":
        gr = pe.get("grid_range", [-3.0, 3.0])
        cmd += [
            f"model.decoder.pair_encoder.spline_order={pe.get('spline_order', 3)}",
            f"model.decoder.pair_encoder.grid_size={pe.get('grid_size', 5)}",
            f"model.decoder.pair_encoder.grid_range=[{gr[0]},{gr[1]}]",
            "model.decoder.pair_encoder.grid_update=false",
            f"model.decoder.pair_encoder.base_activation={pe.get('base_activation', 'silu')}",
            f"model.decoder.pair_encoder.base_scale_init={pe.get('base_scale_init', 1.0)}",
            f"model.decoder.pair_encoder.spline_scale_init={pe.get('spline_scale_init', 0.1)}",
            "model.decoder.pair_encoder.use_bias=true",
            f"model.decoder.pair_encoder.spline_l1={pe.get('spline_l1', 1.0e-5)}",
        ]
    else:
        cmd += [
            f"model.decoder.pair_encoder.activation={pe.get('activation', 'gelu')}",
            "model.decoder.pair_encoder.layernorm=true",
        ]
    return cmd


def run_one(
    model_id: str,
    scenario: str,
    seed: int,
    epochs: int,
    *,
    skip_existing: bool,
) -> dict:
    cfg = yaml.safe_load(MODELS[model_id].read_text())
    run_dir = OUT / "runs" / model_id / scenario / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    ckpt = run_dir / "best_model.pt"
    if skip_existing and metrics_path.exists() and ckpt.exists():
        print(f"SKIP {model_id} {scenario} seed={seed}", flush=True)
        return json.loads(metrics_path.read_text())

    cmd = build_train_cmd(
        model_id=model_id,
        cfg=cfg,
        scenario=scenario,
        seed=seed,
        epochs=epochs,
        run_dir=run_dir,
    )
    print("\nRUN:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_env())
    runtime_s = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Failed {model_id} {scenario} seed={seed} rc={proc.returncode}")

    row = {
        "model": model_id,
        "variant": cfg["variant"],
        "scenario": scenario,
        "seed": seed,
        "runtime_s": runtime_s,
        "peak_memory_mb": _peak_memory_mb(),
        **_metrics_from_ckpt(ckpt),
    }
    metrics_path.write_text(json.dumps(row, indent=2, default=float))

    # Endpoint-path audit for this run
    eval_cmd = [
        PY,
        str(ROOT / "scripts" / "eval_taska_run_endpoint_paths.py"),
        f"--model={model_id}",
        f"--scenario={scenario}",
        f"--seed={seed}",
    ]
    print("EVAL:", " ".join(eval_cmd), flush=True)
    ev = subprocess.run(eval_cmd, cwd=str(ROOT), env=_env())
    if ev.returncode != 0:
        raise SystemExit(f"Endpoint eval failed {model_id} {scenario} seed={seed}")
    return row


def append_csv(path: Path, row: dict, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def summarize(rows: list[dict]) -> None:
    global_path = OUT / "global_metrics.csv"
    fields = [
        "model",
        "variant",
        "scenario",
        "seed",
        "valid_auprc",
        "test_auprc",
        "valid_auroc",
        "test_auroc",
        "valid_brier",
        "test_brier",
        "best_epoch",
        "parameter_count",
        "runtime_s",
        "peak_memory_mb",
        "candidate_fingerprint",
    ]
    if global_path.exists():
        global_path.unlink()
    for r in rows:
        append_csv(global_path, r, fields)

    # Scenario summary mean±SD
    sum_path = OUT / "scenario_summary.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "model",
                "n_seeds",
                "valid_auprc_mean",
                "valid_auprc_sd",
                "test_auprc_mean",
                "test_auprc_sd",
                "delta_valid_kan_minus_mlp",
            ]
        )
        by = {}
        for r in rows:
            by.setdefault((r["scenario"], r["model"]), []).append(r)
        for scenario in SCENARIOS:
            stats = {}
            for model in MODELS:
                rs = by.get((scenario, model), [])
                vm, vs = _mean_sd([float(r["valid_auprc"]) for r in rs])
                tm, ts = _mean_sd([float(r["test_auprc"]) for r in rs])
                stats[model] = (vm, vs, tm, ts, len(rs))
                w.writerow([scenario, model, len(rs), vm, vs, tm, ts, ""])
            if "TA_MLP" in stats and "TA_KAN" in stats:
                d = stats["TA_KAN"][0] - stats["TA_MLP"][0]
                w.writerow([scenario, "DELTA_KAN_MINUS_MLP", "", d, "", "", "", d])

    # Fairness fingerprints: MLP vs KAN same scenario+seed
    fair = []
    keyed = {(r["scenario"], r["seed"], r["model"]): r for r in rows}
    for scenario in SCENARIOS:
        for seed in SEEDS:
            a = keyed.get((scenario, seed, "TA_MLP"))
            b = keyed.get((scenario, seed, "TA_KAN"))
            if not a or not b:
                continue
            fair.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "mlp_fp": a.get("candidate_fingerprint"),
                    "kan_fp": b.get("candidate_fingerprint"),
                    "match": a.get("candidate_fingerprint") == b.get("candidate_fingerprint"),
                }
            )
    (OUT / "fairness_fingerprints.json").write_text(json.dumps(fair, indent=2))

    decision = {
        "experiment": "TASKA_MLP_VS_KAN_FULL",
        "selection_metric": "valid_auprc",
        "n_runs_completed": len(rows),
        "n_runs_expected": len(SCENARIOS) * len(SEEDS) * len(MODELS),
        "models": list(MODELS),
        "scenarios": list(SCENARIOS),
        "seeds": list(SEEDS),
        "note": "Finalize after all 36 runs; Δ_s = mean_valid_KAN - mean_valid_MLP per scenario.",
        "fairness_fingerprint_mismatches": sum(1 for x in fair if not x["match"]),
    }
    # fill per-scenario deltas when both present
    scenario_deltas = {}
    for scenario in SCENARIOS:
        mlp = by.get((scenario, "TA_MLP"), [])
        kan = by.get((scenario, "TA_KAN"), [])
        if mlp and kan:
            m_m, _ = _mean_sd([float(r["valid_auprc"]) for r in mlp])
            m_k, _ = _mean_sd([float(r["valid_auprc"]) for r in kan])
            scenario_deltas[scenario] = {
                "mlp_valid_mean": m_m,
                "kan_valid_mean": m_k,
                "delta_kan_minus_mlp": m_k - m_m,
            }
    decision["scenario_deltas"] = scenario_deltas
    if scenario_deltas:
        mean_d = sum(v["delta_kan_minus_mlp"] for v in scenario_deltas.values()) / len(
            scenario_deltas
        )
        decision["mean_delta_across_scenarios"] = mean_d
        decision["winner_global"] = (
            "TA_KAN" if mean_d > 0.005 else ("TA_MLP" if mean_d < -0.005 else "tie_band")
        )
    (OUT / "FINAL_DECISION.json").write_text(json.dumps(decision, indent=2, default=float))


def _require_context_registry() -> None:
    if CONTEXT_REGISTRY.exists():
        return
    raise SystemExit(
        f"Missing Wave 5C edge-context registry: {CONTEXT_REGISTRY}\n"
        "Build it first (requires Wave 5 evidence under outputs/wave5/evidence/):\n"
        f"  PYTHONPATH=src {PY} {ROOT / 'scripts' / 'build_wave5c_context_registry.py'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--models", default="all", help="TA_MLP,TA_KAN or all")
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--build-registry", action="store_true", default=True)
    args = ap.parse_args()

    _require_context_registry()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    if args.build_registry:
        proc = subprocess.run(
            [PY, str(ROOT / "scripts" / "build_taska_edge_endpoint_registry.py")],
            cwd=str(ROOT),
            env=_env(),
        )
        if proc.returncode != 0:
            raise SystemExit("registry build failed")

    scenarios = list(SCENARIOS) if args.scenarios == "all" else [
        s.strip() for s in args.scenarios.split(",") if s.strip()
    ]
    seeds = list(SEEDS) if args.seeds == "all" else [
        int(s.strip()) for s in args.seeds.split(",") if s.strip()
    ]
    models = list(MODELS) if args.models == "all" else [
        m.strip() for m in args.models.split(",") if m.strip()
    ]
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"Unknown model {m}")

    # Manifest
    manifest = []
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            for model in models:
                manifest.append({"model": model, "scenario": scenario, "seed": seed})
                row = run_one(
                    model,
                    scenario,
                    seed,
                    args.epochs,
                    skip_existing=not args.no_skip,
                )
                rows.append(row)
                append_csv(OUT / "run_manifest.csv", {**manifest[-1], "status": "done"})

    summarize(rows)

    # Aggregate endpoint CSVs written per-run
    ep_parts = list(OUT.glob("runs/*/*/*/endpoint_path_metrics.csv"))
    if ep_parts:
        import pandas as pd

        frames = [pd.read_csv(p) for p in ep_parts]
        all_ep = pd.concat(frames, ignore_index=True)
        all_ep.to_csv(OUT / "endpoint_path_metrics.csv", index=False)
        # summary mean over seeds
        gcols = ["scenario", "model", "endpoint", "split"]
        num_cols = [
            c
            for c in all_ep.columns
            if c not in gcols + ["seed", "hospital"]
            and pd.api.types.is_numeric_dtype(all_ep[c])
        ]
        summary = (
            all_ep.groupby(gcols, dropna=False)[num_cols].agg(["mean", "std"]).reset_index()
        )
        summary.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns
        ]
        summary.to_csv(OUT / "endpoint_path_summary.csv", index=False)

        mh = all_ep[all_ep["scenario"] == "multihospital"].copy()
        if len(mh):
            mh.to_csv(OUT / "multihospital_endpoint_metrics.csv", index=False)

    print(f"Done. outputs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
