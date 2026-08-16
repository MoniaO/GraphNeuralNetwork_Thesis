#!/usr/bin/env python3
"""Edge × probability × clinical_pathway report for Stage C / FINAL S10.

Joins candidate predictions to audited GSN v3 edges (clinical_pathway,
edge_type, dag_motif) and aggregates rare vs common pathway buckets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CONFIG_DIR = str(REPO / "configs")
DEFAULT_GSN = Path.home() / "Desktop" / "GSN Graphs dysertation 2026"
CANDIDATE_SEED = 20260722


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def load_edge_meta(gsn: Path) -> pd.DataFrame:
    path = (
        gsn
        / "2 v3. Data"
        / "dataset_v3"
        / "synthetic_pharmacotherapy_v3_edges_audited.csv"
    )
    df = pd.read_csv(path)
    keep = [
        c
        for c in (
            "source",
            "target",
            "edge_type",
            "dag_motif",
            "clinical_pathway",
            "activation_frequency",
        )
        if c in df.columns
    ]
    return df[keep].drop_duplicates(subset=["source", "target"])


def build_overrides(
    scenario: str,
    seed: int,
    *,
    encoder: str = "mlp",
) -> list[str]:
    from taskA_FINAL_14_08_2026.runner import build_overrides as bo

    return bo(encoder=encoder, scenario=scenario, training_seed=seed)  # type: ignore[arg-type]


@torch.no_grad()
def predict(model, data, device) -> pd.DataFrame:
    model.eval()
    data = data.to(device)
    logits = model(data).view(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.float().cpu().numpy()
    src = list(getattr(data, "candidate_source_name", []))
    tgt = list(getattr(data, "candidate_target_name", []))
    if not src:
        # fallback indices only
        src = [str(i) for i in range(len(probs))]
        tgt = [str(i) for i in range(len(probs))]
    return pd.DataFrame(
        {
            "source": src,
            "target": tgt,
            "label": labels.astype(int),
            "probability": probs.astype(float),
        }
    )


def prevalence_bucket(n_pos: int, n: int) -> str:
    if n <= 0:
        return "empty"
    prev = n_pos / n
    if prev < 0.05:
        return "rare"
    if prev < 0.15:
        return "mid"
    return "common"


def aggregate_pathway(df: pd.DataFrame, col: str = "clinical_pathway") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col, dropna=False):
        y = g["label"].to_numpy()
        p = g["probability"].to_numpy()
        n = len(g)
        n_pos = int(y.sum())
        rows.append(
            {
                col: key,
                "n": n,
                "n_pos": n_pos,
                "prevalence": n_pos / n if n else float("nan"),
                "bucket": prevalence_bucket(n_pos, n),
                "auprc": _auprc(y, p),
                "auc": _auc(y, p),
                "mean_prob_pos": float(p[y == 1].mean()) if n_pos else float("nan"),
                "mean_prob_neg": float(p[y == 0].mean()) if (n - n_pos) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["bucket", "prevalence", "n"], ascending=[True, True, False])
    return out


def eval_one_ckpt(
    ckpt: Path,
    *,
    scenario: str,
    seed: int,
    encoder: str,
    edge_meta: pd.DataFrame,
    out_dir: Path,
    tag: str,
) -> dict[str, Any]:
    from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
    from taskA_final_large_grid_11_08_2026.stage_c.attach import (
        fit_and_attach_stage_c_stats,
    )
    from train_taskA import build_model

    overrides = build_overrides(scenario, seed, encoder=encoder)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    train_data, valid_data, test_data, _ = load_recon_heterodata(cfg)
    fit_and_attach_stage_c_stats(cfg, train_data, valid_data, test_data, device="cpu")
    model = build_model(cfg, train_data)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    device = torch.device("cpu")
    model.to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"tag": tag, "scenario": scenario, "seed": seed, "encoder": encoder}
    for split, data in (("valid", valid_data), ("test", test_data)):
        pred = predict(model, data, device)
        merged = pred.merge(edge_meta, on=["source", "target"], how="left")
        merged["split"] = split
        merged["scenario"] = scenario
        merged["seed"] = seed
        merged["encoder"] = encoder
        merged["model_tag"] = tag
        csv_path = out_dir / f"predictions_{split}_{tag}.csv"
        merged.to_csv(csv_path, index=False)

        y = merged["label"].to_numpy()
        p = merged["probability"].to_numpy()
        summary[f"{split}_auprc"] = _auprc(y, p)
        summary[f"{split}_auc"] = _auc(y, p)
        summary[f"{split}_n"] = int(len(merged))
        summary[f"{split}_coverage_pathway"] = float(
            merged["clinical_pathway"].notna().mean()
        )

        for col in ("clinical_pathway", "edge_type", "dag_motif"):
            if col not in merged.columns:
                continue
            agg = aggregate_pathway(merged, col=col)
            agg.to_csv(out_dir / f"{col}_metrics_{split}_{tag}.csv", index=False)

        # rare bucket rollup on clinical_pathway
        if "clinical_pathway" in merged.columns:
            tmp = merged.copy()
            # assign bucket by global pathway prevalence on this split
            prev = (
                tmp.groupby("clinical_pathway")["label"]
                .agg(["sum", "count"])
                .reset_index()
            )
            prev["bucket"] = [
                prevalence_bucket(int(s), int(c))
                for s, c in zip(prev["sum"], prev["count"])
            ]
            tmp = tmp.merge(
                prev[["clinical_pathway", "bucket"]], on="clinical_pathway", how="left"
            )
            bucket_rows = []
            for b, g in tmp.groupby("bucket"):
                yy = g["label"].to_numpy()
                pp = g["probability"].to_numpy()
                bucket_rows.append(
                    {
                        "split": split,
                        "bucket": b,
                        "n": len(g),
                        "n_pos": int(yy.sum()),
                        "auprc": _auprc(yy, pp),
                        "auc": _auc(yy, pp),
                    }
                )
            pd.DataFrame(bucket_rows).to_csv(
                out_dir / f"rare_bucket_{split}_{tag}.csv", index=False
            )
            for br in bucket_rows:
                summary[f"{split}_{br['bucket']}_auprc"] = br["auprc"]
                summary[f"{split}_{br['bucket']}_n"] = br["n"]

    (out_dir / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def discover_11_08() -> list[tuple[Path, str, int]]:
    root = REPO / "outputs/taskA_final_large_grid_11.08.2026/stage_c/runs"
    found = []
    for ckpt in sorted(root.rglob("S10_HCR_FULL40/*/best_model.pt")):
        # .../runs/{scenario}/hd4/S10_HCR_FULL40/seed{s}/best_model.pt
        seed_s = ckpt.parent.name.replace("seed", "")
        scenario = ckpt.parents[3].name
        found.append((ckpt, scenario, int(seed_s)))
    return found


def discover_final(encoder: str) -> list[tuple[Path, str, int]]:
    root = REPO / "outputs/taskA_FINAL_14.08.2026/runs" / encoder
    found = []
    if not root.exists():
        return found
    for ckpt in sorted(root.rglob("best_model.pt")):
        # .../runs/{encoder}/{scenario}/S10_HCR_FULL40/seed{s}/best_model.pt
        seed_s = ckpt.parent.name.replace("seed", "")
        scenario = ckpt.parents[2].name
        found.append((ckpt, scenario, int(seed_s)))
    return found


def write_rare_md(summaries: list[dict], path: Path) -> None:
    lines = [
        "# Rare pathway report — FINAL 14.08.2026\n",
        "Buckets by pathway prevalence on the eval split: "
        "`rare` (<5% pos), `mid` (5–15%), `common` (≥15%).\n",
        "| tag | scenario | seed | valid rare AUPRC | valid mid | valid common | "
        "test rare | coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| `{s.get('tag')}` | {s.get('scenario')} | {s.get('seed')} | "
            f"{s.get('valid_rare_auprc', float('nan')):.3f} | "
            f"{s.get('valid_mid_auprc', float('nan')):.3f} | "
            f"{s.get('valid_common_auprc', float('nan')):.3f} | "
            f"{s.get('test_rare_auprc', float('nan')):.3f} | "
            f"{s.get('valid_coverage_pathway', float('nan')):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["11.08", "final", "both"], default="11.08")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--encoder", default="mlp", choices=["mlp", "kan_shallow"])
    args = ap.parse_args()

    os.environ.setdefault("GSN_PROJECT_ROOT", str(DEFAULT_GSN))
    gsn = Path(os.environ["GSN_PROJECT_ROOT"])
    edge_meta = load_edge_meta(gsn)

    jobs: list[tuple[Path, str, int, str, str]] = []
    # (ckpt, scenario, seed, encoder, out_subdir)
    if args.source in {"11.08", "both"}:
        for ckpt, sc, seed in discover_11_08():
            jobs.append((ckpt, sc, seed, "mlp", "from_11.08_S10"))
    if args.source in {"final", "both"}:
        for enc in ("mlp", "kan_shallow"):
            for ckpt, sc, seed in discover_final(enc):
                jobs.append((ckpt, sc, seed, enc, f"FINAL_{enc}"))

    if args.max_runs is not None:
        jobs = jobs[: args.max_runs]

    summaries = []
    for i, (ckpt, sc, seed, enc, sub) in enumerate(jobs, 1):
        tag = f"{enc}__{sc}__seed{seed}"
        out_dir = (
            REPO
            / "outputs/taskA_FINAL_14.08.2026/edge_pathway_report"
            / sub
        )
        print(f"[{i}/{len(jobs)}] {tag}", flush=True)
        try:
            summaries.append(
                eval_one_ckpt(
                    ckpt,
                    scenario=sc,
                    seed=seed,
                    encoder=enc if sub.startswith("FINAL") else "mlp",
                    edge_meta=edge_meta,
                    out_dir=out_dir,
                    tag=tag,
                )
            )
        except Exception as e:
            print(f"FAIL {tag}: {e}", flush=True)
            summaries.append(
                {"tag": tag, "scenario": sc, "seed": seed, "encoder": enc, "error": str(e)}
            )

    out_root = REPO / "outputs/taskA_FINAL_14.08.2026/edge_pathway_report"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ALL_SUMMARIES_14.08.2026.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    write_rare_md(summaries, out_root / "rare_pathway_table.md")
    print(f"Wrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
