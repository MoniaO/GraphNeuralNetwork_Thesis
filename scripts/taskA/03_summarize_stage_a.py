#!/usr/bin/env python3
"""03 — tabele decyzji Stage A (po 01). Selekcja: mean valid AUPRC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = (
    REPO
    / "outputs"
    / "taskA_final_large_grid_11.08.2026"
    / "stage_a"
)
RUNS = ROOT / "runs" / "shared"


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], float("nan")
    return statistics.mean(xs), statistics.stdev(xs)


def collect(scenario: str) -> list[dict]:
    base = RUNS / scenario
    rows = []
    for p in sorted(base.rglob("result_11.08.2026.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        cfg_id = p.parent.parent.name
        seed = p.parent.name.replace("seed", "")
        sel = d.get("selection") or {}
        sealed = d.get("sealed_test") or {}
        parts = cfg_id.split("__")
        backbone = parts[0]
        rows.append(
            {
                "config_id": cfg_id,
                "backbone": backbone,
                "seed": int(seed),
                "status": d.get("status"),
                "valid_auprc": sel.get("valid_auprc"),
                "valid_auc": sel.get("valid_auc"),
                "valid_brier": sel.get("valid_brier"),
                "best_epoch": sel.get("best_epoch"),
                "sealed_test_auprc": sealed.get("test_auprc"),
                "sealed_test_auc": sealed.get("test_auc"),
                "candidate_seed": d.get("candidate_seed"),
                "candidate_fingerprint": d.get("candidate_fingerprint"),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("valid_auprc") is None:
            continue
        by_cfg[r["config_id"]].append(r)

    out = []
    for cfg_id, xs in by_cfg.items():
        auprcs = [float(x["valid_auprc"]) for x in xs]
        aucs = [float(x["valid_auc"]) for x in xs if x.get("valid_auc") is not None]
        sealed = [
            float(x["sealed_test_auprc"])
            for x in xs
            if x.get("sealed_test_auprc") is not None
        ]
        m, s = _mean_std(auprcs)
        am, as_ = _mean_std(aucs)
        sm, ss = _mean_std(sealed)
        out.append(
            {
                "config_id": cfg_id,
                "backbone": xs[0]["backbone"],
                "n_seeds": len(xs),
                "mean_valid_auprc": m,
                "std_valid_auprc": s,
                "mean_valid_auc": am,
                "std_valid_auc": as_,
                "mean_sealed_test_auprc": sm,
                "std_sealed_test_auprc": ss,
                "seeds": ",".join(str(x["seed"]) for x in sorted(xs, key=lambda z: z["seed"])),
            }
        )
    out.sort(key=lambda r: (-(r["mean_valid_auprc"] or -1), r["config_id"]))
    return out


def fmt(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def write_markdown(
    agg: list[dict],
    *,
    scenario: str,
    n_ok: int,
    n_fail: int,
    n_total: int,
    path: Path,
) -> None:
    complete = n_ok + n_fail >= n_total and n_fail == 0
    lines = [
        f"# Stage A backbone decision table — 11.08.2026",
        "",
        f"- Scenario screen: `{scenario}`",
        f"- Progress: **{n_ok} ok / {n_fail} fail / {n_total} planned**"
        + (" — COMPLETE" if complete else " — IN PROGRESS"),
        "- Selection metric: **mean valid AUPRC** over seeds (test sealed, not ranked)",
        "- Next stop: pick Top-1 (or Top-2) config **per backbone** before Stage C stats layer",
        "",
        "## Overall ranking (valid AUPRC)",
        "",
        "| rank | backbone | config_id | n_seeds | mean valid AUPRC ± std | mean valid AUC ± std | sealed test AUPRC (info only) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(agg, 1):
        lines.append(
            "| {rank} | {bb} | `{cfg}` | {n} | {ma} ± {sa} | {mau} ± {sau} | {st} ± {sts} |".format(
                rank=i,
                bb=r["backbone"],
                cfg=r["config_id"],
                n=r["n_seeds"],
                ma=fmt(r["mean_valid_auprc"]),
                sa=fmt(r["std_valid_auprc"]),
                mau=fmt(r["mean_valid_auc"]),
                sau=fmt(r["std_valid_auc"]),
                st=fmt(r["mean_sealed_test_auprc"]),
                sts=fmt(r["std_sealed_test_auprc"]),
            )
        )

    lines += ["", "## Top-2 per backbone (selection shortlist)", ""]
    by_bb: dict[str, list[dict]] = defaultdict(list)
    for r in agg:
        by_bb[r["backbone"]].append(r)
    for bb in sorted(by_bb):
        lines.append(f"### {bb}")
        lines.append("")
        lines.append(
            "| rank | config_id | mean valid AUPRC ± std | n_seeds |"
        )
        lines.append("| ---: | --- | ---: | ---: |")
        for i, r in enumerate(by_bb[bb][:2], 1):
            lines.append(
                f"| {i} | `{r['config_id']}` | {fmt(r['mean_valid_auprc'])} ± {fmt(r['std_valid_auprc'])} | {r['n_seeds']} |"
            )
        lines.append("")

    if agg:
        winner = agg[0]
        lines += [
            "## Suggested freeze candidate (clean screen only)",
            "",
            f"- Best mean valid AUPRC: `{winner['config_id']}` "
            f"({fmt(winner['mean_valid_auprc'])} ± {fmt(winner['std_valid_auprc'])}, "
            f"n={winner['n_seeds']})",
            "- Protocol next: Top-2 / backbone → 6 scenarios × 3 seeds, then freeze one / backbone",
            "- **Do not** use sealed test for this choice",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="clean")
    p.add_argument(
        "--total",
        type=int,
        default=0,
        help="Planned jobs (0 = count from stage_a_grid)",
    )
    args = p.parse_args()

    total = int(args.total)
    if total <= 0:
        sys.path.insert(0, str(REPO / "src"))
        from taskA.experiments.stage_a_backbone.grid import (  # noqa: WPS433
            count_shared_jobs,
        )

        total = int(count_shared_jobs(3))

    rows = collect(args.scenario)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_fail = sum(1 for r in rows if r["status"] != "ok")
    agg = aggregate(rows)

    ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "STAGE_A_BACKBONE_RANKING_11.08.2026.csv"
    md_path = ROOT / "STAGE_A_BACKBONE_DECISION_TABLE_11.08.2026.md"
    seed_csv = ROOT / "STAGE_A_BACKBONE_PER_SEED_11.08.2026.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "backbone",
                "config_id",
                "n_seeds",
                "mean_valid_auprc",
                "std_valid_auprc",
                "mean_valid_auc",
                "std_valid_auc",
                "mean_sealed_test_auprc",
                "std_sealed_test_auprc",
                "seeds",
            ],
        )
        w.writeheader()
        for i, r in enumerate(agg, 1):
            w.writerow({"rank": i, **r})

    with seed_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "backbone",
                "config_id",
                "seed",
                "status",
                "valid_auprc",
                "valid_auc",
                "valid_brier",
                "best_epoch",
                "sealed_test_auprc",
                "sealed_test_auc",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    write_markdown(
        agg,
        scenario=args.scenario,
        n_ok=n_ok,
        n_fail=n_fail,
        n_total=total,
        path=md_path,
    )
    print(f"ok={n_ok} fail={n_fail} configs_ranked={len(agg)}")
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    if agg:
        print(
            f"leader: {agg[0]['config_id']}  "
            f"mean_valid_auprc={agg[0]['mean_valid_auprc']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
