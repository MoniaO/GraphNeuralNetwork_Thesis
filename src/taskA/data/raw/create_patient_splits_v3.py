#!/usr/bin/env python3
"""Create deterministic, multilabel-balanced v3 patient splits."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA = PROJECT_ROOT / "2 v3. Data" / "dataset_v3"
OUTPUT = PROJECT_ROOT / "2 v3. Data" / "splits"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v3_interaction_spec import ENDPOINTS  # noqa: E402

SCENARIOS = [
    "clean", "hidden_confounder", "selection_bias", "no_overlap",
    "noisy_documentation", "multihospital",
]
FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SEED = 20260722


def stable_hash(patient_id: str, split: str = "") -> int:
    value = f"{SEED}|{patient_id}|{split}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"patient_id", *ENDPOINTS}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return rows


def assign(rows: list[dict[str, str]]) -> dict[str, str]:
    n = len(rows)
    capacities = {
        "train": int(n * 0.70),
        "validation": int(n * 0.15),
        "test": n - int(n * 0.70) - int(n * 0.15),
    }
    totals = {
        endpoint: sum(int(float(row[endpoint])) for row in rows)
        for endpoint in ENDPOINTS
    }
    desired = {
        (endpoint, split): totals[endpoint] * FRACTIONS[split]
        for endpoint in ENDPOINTS
        for split in FRACTIONS
    }
    sizes: Counter[str] = Counter()
    positives: dict[tuple[str, str], int] = defaultdict(int)

    def rarity(row: dict[str, str]) -> float:
        return sum(
            int(float(row[endpoint])) / max(totals[endpoint], 1)
            for endpoint in ENDPOINTS
        )

    ordered = sorted(
        rows, key=lambda row: (-rarity(row), stable_hash(row["patient_id"]))
    )
    assignments: dict[str, str] = {}
    for row in ordered:
        patient_id = row["patient_id"]
        labels = [
            endpoint for endpoint in ENDPOINTS
            if int(float(row[endpoint])) == 1
        ]
        eligible = [
            split for split in FRACTIONS if sizes[split] < capacities[split]
        ]

        def score(split: str) -> tuple[float, float, int]:
            label_need = sum(
                (desired[(endpoint, split)] - positives[(endpoint, split)])
                / max(desired[(endpoint, split)], 1.0)
                for endpoint in labels
            )
            capacity_need = (
                capacities[split] - sizes[split]
            ) / capacities[split]
            return label_need, capacity_need, -stable_hash(patient_id, split)

        selected = max(eligible, key=score)
        assignments[patient_id] = selected
        sizes[selected] += 1
        for endpoint in labels:
            positives[(endpoint, selected)] += 1
    if sizes != Counter(capacities):
        raise RuntimeError(f"Incorrect split sizes: {sizes} != {capacities}")
    return assignments


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clean = load_rows(DATA / "synthetic_pharmacotherapy_v3_samples_clean.csv")
    assignments = assign(clean)

    assignment_path = OUTPUT / "patient_splits_v3.csv"
    order = {"train": 0, "validation": 1, "test": 2}
    with assignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patient_id", "split", "split_seed", "reference_scenario"],
        )
        writer.writeheader()
        for patient_id, split in sorted(
            assignments.items(), key=lambda item: (order[item[1]], int(item[0]))
        ):
            writer.writerow({
                "patient_id": patient_id,
                "split": split,
                "split_seed": SEED,
                "reference_scenario": "clean",
            })

    summary: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        rows = load_rows(
            DATA / f"synthetic_pharmacotherapy_v3_samples_{scenario}.csv"
        )
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[assignments[row["patient_id"]]].append(row)
        for split in FRACTIONS:
            split_rows = grouped[split]
            record: dict[str, object] = {
                "scenario": scenario, "split": split, "n_rows": len(split_rows)
            }
            for endpoint in ENDPOINTS:
                count = sum(int(float(row[endpoint])) for row in split_rows)
                record[f"{endpoint}_n"] = count
                record[f"{endpoint}_rate"] = (
                    count / len(split_rows) if split_rows else 0.0
                )
            summary.append(record)

    with (OUTPUT / "patient_split_summary_v3.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    counts = Counter(assignments.values())
    report = {
        "status": "PASS",
        "seed": SEED,
        "strategy": "deterministic greedy multilabel balancing",
        "reference_scenario": "clean",
        "endpoints": ENDPOINTS,
        "n_unique_patients": len(assignments),
        "split_counts": dict(counts),
        "all_clean_patients_assigned_once": len(assignments) == len(clean),
        "selection_bias_inherits_master_assignments": True,
    }
    (OUTPUT / "patient_split_validation_v3.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
