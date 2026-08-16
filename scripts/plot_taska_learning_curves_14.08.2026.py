#!/usr/bin/env python3
"""Parse Task A train logs → epoch CSV + PNG learning curves (FINAL 14.08.2026)."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+optim\s+([0-9.nan]+)\s+\|\s+train AUPRC\s+([0-9.nan]+)"
    r"\s+\|\s+valid AUPRC\s+([0-9.nan]+)\s+\|\s+valid AUC\s+([0-9.nan]+)"
    r"\s+\|\s+best\s+(\d+)"
)


def parse_log(path: Path) -> list[dict]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        ep, loss, tr, va, auc, best = m.groups()
        rows.append(
            {
                "epoch": int(ep),
                "loss": float(loss),
                "train_auprc": float(tr),
                "valid_auprc": float(va),
                "valid_auc": float(auc),
                "best_epoch": int(best),
                "gap_train_valid": float(tr) - float(va),
            }
        )
    return rows


def plot_run(rows: list[dict], out_png: Path, title: str) -> None:
    if not rows:
        return
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(ep, [r["loss"] for r in rows], label="optim loss", color="C0")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, [r["train_auprc"] for r in rows], label="train AUPRC", color="C1")
    ax.plot(ep, [r["valid_auprc"] for r in rows], label="valid AUPRC", color="C2")
    be = rows[-1]["best_epoch"]
    ax.axvline(be, color="k", ls="--", alpha=0.5, label=f"best={be}")
    ax.set_title("AUPRC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ep, [r["valid_auc"] for r in rows], color="C3")
    ax.set_title("Valid AUC")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ep, [r["gap_train_valid"] for r in rows], color="C4")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Gap train−valid AUPRC")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_mean_panel(
    by_key: dict[str, list[list[dict]]], out_png: Path, title: str
) -> None:
    """Mean±std valid AUPRC across seeds; keys are labels."""
    if not by_key:
        return
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for label, runs in sorted(by_key.items()):
        max_ep = max(len(r) for r in runs if r)
        if max_ep == 0:
            continue
        mat = np.full((len(runs), max_ep), np.nan)
        for i, rows in enumerate(runs):
            for j, r in enumerate(rows[:max_ep]):
                mat[i, j] = r["valid_auprc"]
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        xs = np.arange(1, max_ep + 1)
        ax.plot(xs, mean, label=label)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid AUPRC")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def collect_from_11_08(root: Path) -> list[tuple[str, Path, list[dict]]]:
    out = []
    base = root / "outputs/taskA_final_large_grid_11.08.2026/stage_c/runs"
    for log in sorted(base.rglob("train_11.08.2026.log")):
        if "S10_HCR_FULL40" not in str(log):
            continue
        rows = parse_log(log)
        # scenario/hd4/S10/.../seed
        parts = log.parts
        try:
            i = parts.index("runs")
            scenario = parts[i + 1]
            seed = parts[i + 4]
        except Exception:
            scenario, seed = "?", "?"
        label = f"11.08_S10_{scenario}_{seed}"
        out.append((label, log, rows))
    return out


def collect_from_final(root: Path) -> list[tuple[str, Path, list[dict]]]:
    out = []
    base = root / "outputs/taskA_FINAL_14.08.2026/runs"
    if not base.exists():
        return out
    for log in sorted(base.rglob("train_14.08.2026.log")):
        rows = parse_log(log)
        parts = log.parts
        try:
            i = parts.index("runs")
            enc, scenario, _var, seed = parts[i + 1 : i + 5]
        except Exception:
            enc, scenario, seed = "?", "?", "?"
        label = f"FINAL_{enc}_{scenario}_{seed}"
        out.append((label, log, rows))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["11.08", "final", "both"],
        default="both",
    )
    args = ap.parse_args()

    out_root = REPO / "outputs/taskA_FINAL_14.08.2026/learning_curves"
    items: list[tuple[str, Path, list[dict]]] = []
    if args.source in {"11.08", "both"}:
        items.extend(collect_from_11_08(REPO))
    if args.source in {"final", "both"}:
        items.extend(collect_from_final(REPO))

    all_csv = out_root / "all_epoch_histories.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "epoch",
                "loss",
                "train_auprc",
                "valid_auprc",
                "valid_auc",
                "best_epoch",
                "gap_train_valid",
            ],
        )
        w.writeheader()
        by_scen: dict[str, list[list[dict]]] = defaultdict(list)
        for label, log, rows in items:
            if not rows:
                continue
            sub = out_root / ("from_11.08_S10" if label.startswith("11.08") else "FINAL")
            plot_run(rows, sub / f"{label}.png", title=label)
            hist = sub / f"{label}_epochs.csv"
            with hist.open("w", newline="", encoding="utf-8") as hf:
                hw = csv.DictWriter(hf, fieldnames=list(rows[0].keys()))
                hw.writeheader()
                hw.writerows(rows)
            for r in rows:
                w.writerow({"run": label, **r})
            # group for mean panel: 11.08 by scenario; FINAL by enc+scenario
            if label.startswith("11.08"):
                sc = label.split("_")[2]
                by_scen[f"11.08/{sc}"].append(rows)
            else:
                # FINAL_mlp_clean_seed...
                bits = label.split("_")
                key = f"{bits[1]}/{bits[2]}"
                by_scen[key].append(rows)

    plot_mean_panel(
        by_scen,
        out_root / "mean_valid_auprc_by_group.png",
        "Mean±std valid AUPRC",
    )

    # diagnostics md
    lines = [
        "# Learning diagnostics 14.08.2026\n",
        f"Parsed runs: **{len(items)}** (with epochs: {sum(1 for _,_,r in items if r)}).\n",
        "| run | n_ep | best_ep | best_valid | final_train | final_gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _, rows in items:
        if not rows:
            continue
        best = max(rows, key=lambda r: r["valid_auprc"])
        last = rows[-1]
        lines.append(
            f"| `{label}` | {len(rows)} | {best['epoch']} | {best['valid_auprc']:.3f} "
            f"| {last['train_auprc']:.3f} | {last['gap_train_valid']:.3f} |"
        )
    (out_root / "LEARNING_DIAGNOSTICS_14.08.2026.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote curves under {out_root} ({len(items)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
