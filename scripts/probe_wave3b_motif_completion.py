#!/usr/bin/env python3
"""Wave 3B diagnostic probe: does HCR-3 a111 recover dual-gate AND structure?

This is NOT the full GNN motif-completion train yet. It answers the narrower
question: on train patients only, do third-order features separate true
parent→gate edges from matched non-edges / shuffled-Z placebos?

Writes AUCs over motif-completion positives vs hard negatives.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
from hcr.binary_features import binary_pair_features, result_to_vector
from hcr.motifs import (
    export_motif_catalog,
    extract_dual_gates,
    load_truth_graph,
    motif_completion_tasks,
)
from hcr.triple_features import (
    binary_triple_features,
    shuffled_z_triple_features,
    triple_result_to_vector,
)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                "model=TaskA_hgt",
                "hcr=none",
                "wandb.enabled=false",
                "data.dataset.scenario=clean",
            ],
        )


def main() -> None:
    day = date.today().isoformat()
    out = ROOT / "outputs" / f"wave3b_motifs_{day}"
    out.mkdir(parents=True, exist_ok=True)
    export_motif_catalog(out)

    cfg = _cfg()
    OmegaConf.resolve(cfg)
    patients = train_patient_df(load_patient_matrix_with_split(cfg))
    _, edges = load_truth_graph(cfg)
    duals = [g for g in extract_dual_gates(edges, patients) if g.binary_parents]
    tasks = motif_completion_tasks(duals)

    edge_set = set(zip(edges["source"].astype(str), edges["target"].astype(str)))

    # Positives: hidden parent→gate tasks
    positives = [(t.candidate_source, t.candidate_target, t.visible_parent) for t in tasks]

    # Hard negatives: same types / random non-edges among binary nodes used in gates
    parents = sorted({p for t in tasks for p in (t.hidden_parent, t.visible_parent)})
    gates = sorted({t.gate for t in tasks})
    rng = np.random.default_rng(20260722)
    negatives = []
    attempts = 0
    while len(negatives) < len(positives) and attempts < 20000:
        attempts += 1
        a = str(rng.choice(parents))
        g = str(rng.choice(gates))
        z = str(rng.choice([p for p in parents if p != a]))
        if (a, g) in edge_set:
            continue
        if any(p[0] == a and p[1] == g for p in positives):
            continue
        negatives.append((a, g, z))

    rows = []
    for label, triples in ((1, positives), (0, negatives)):
        for src, tgt, z in triples:
            x = patients[src].to_numpy()
            y = patients[tgt].to_numpy()
            zz = patients[z].to_numpy()
            try:
                pair = result_to_vector(binary_pair_features(x, y))
                full = triple_result_to_vector(binary_triple_features(x, y, zz))
                no_a = triple_result_to_vector(
                    binary_triple_features(x, y, zz), include_a111=False
                )
                shuf = triple_result_to_vector(
                    shuffled_z_triple_features(x, y, zz, seed=20260722)
                )
            except ValueError:
                continue
            rows.append(
                {
                    "label": label,
                    "source": src,
                    "target": tgt,
                    "z": z,
                    "hcr2": pair,
                    "hcr3_full": full,
                    "hcr3_without_a111": no_a,
                    "hcr3_shuffled": shuf,
                    "a111": float(full[9]),
                }
            )

    frame = pd.DataFrame(rows)
    detail = out / f"{day}_wave3b_motif_probe_pairs.csv"
    frame.drop(columns=["hcr2", "hcr3_full", "hcr3_without_a111", "hcr3_shuffled"]).to_csv(
        detail, index=False
    )

    y = frame["label"].to_numpy()
    results = []
    for name in ("hcr2", "hcr3_full", "hcr3_without_a111", "hcr3_shuffled"):
        X = np.stack(frame[name].to_numpy())
        # Univariate a111 for full only
        if name == "hcr3_full":
            a = frame["a111"].to_numpy().reshape(-1, 1)
            results.append(
                {
                    "features": "a111_only",
                    "auprc": float(average_precision_score(y, a.ravel())),
                    "auroc": float(roc_auc_score(y, a.ravel())),
                }
            )
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        # tiny set → use in-sample decision scores carefully; prefer CV if n large enough
        if len(frame) >= 20:
            scores = cross_val_predict(clf, X, y, cv=min(5, y.sum(), (1 - y).sum()), method="predict_proba")[
                :, 1
            ]
        else:
            clf.fit(X, y)
            scores = clf.predict_proba(X)[:, 1]
        results.append(
            {
                "features": name,
                "auprc": float(average_precision_score(y, scores)),
                "auroc": float(roc_auc_score(y, scores)),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
            }
        )

    summary = pd.DataFrame(results)
    summary_path = out / f"{day}_wave3b_motif_probe_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"n dual binary gates={len(duals)} tasks={len(tasks)} probe_rows={len(frame)}")


if __name__ == "__main__":
    main()
