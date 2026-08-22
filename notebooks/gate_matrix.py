#!/usr/bin/env python3
"""Macierz bram HCR (endpoint x typ rodzica) z eksportu W&B.

CO TO POKAZUJE. W trybie typed_concat kazdy blok dowodowy jest mnozony przez
uczony skalar hcr_type_gate[endpoint, typ]. Brama startuje od 1.0 i model sam
decyduje, na ile ufa dowodowi z rodzicow danego typu przy danym endpoincie.
Wartosc bliska zeru = "ten kanal nic nie wnosi".

Wynikowa macierz jest interpretowalna: jesli uporzadkowanie bram odtwarza
strukture DAG-u (typy o szerszym pokryciu dostaja wyzsze wagi), model odzyskal
strukture przyczynowa z samego gradientu, bez podpowiedzi.

UWAGA. Blok trojek (G -> P -> E) NIE ma bramy w obecnej implementacji - jest
doklejany po petli po typach, bez mnoznika. Macierz obejmuje wiec 4 typy
rodzicow, nie 5 blokow. Zeby dostac piata kolumne, trzeba rozszerzyc
hcr_type_gate do [n_targets, n_types + 1] i przemnozyc embedding trojek.

Uzycie:
    python gate_matrix.py --runs all_selected_..._opt3.csv --out rysunki/
    python gate_matrix.py --runs plik.csv --run-filter _L3_ --nodes nodes.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TYPE_LABEL = {
    "patient_context": "patient\ncontext",
    "drug_exposure": "drug\nexposure",
    "mechanism": "mechanism",
    "adr_or_intermediate_state": "adr /\nintermediate",
}


def extract(runs_csv: path, run_filter: str | None) -> tuple[pd.DataFrame, str]:
    d = pd.read_csv("/Users/monika/GraphNeuralNetwork_Thesis/outputs/wandb_exports/all_selected_runs_history_exnoisy_transformer_HCR_new_approach_opt3.csv", low_memory=False)
    gate_cols = [c for c in d.columns if c.startswith("hcr_gate/") and "abs" not in c]
    if not gate_cols:
        raise SystemExit("Brak kolumn hcr_gate/ - przebieg bez typed_concat albo logowanie wylaczone.")

    names = [r for r in d.run_name.unique()
             if "typed_concat" in r and (run_filter is None or run_filter in r)]
    if not names:
        raise SystemExit(f"Brak przebiegu typed_concat pasujacego do {run_filter!r}.")
    if len(names) > 1:
        print(f"UWAGA: pasuje {len(names)} przebiegow, biore pierwszy:\n  " + "\n  ".join(names))
    run = names[0]

    g = d[d.run_name == run].sort_values("_step")
    # ostatni wiersz z kompletem wartosci = stan po zakonczeniu treningu
    last = g[gate_cols].dropna(how="all").iloc[-1]

    rows = {}
    for c in gate_cols:
        _, ep, nt = c.split("/", 2)
        rows.setdefault(ep, {})[nt] = float(last[c])
    return pd.DataFrame(rows).T, run


def add_coverage(mat: pd.DataFrame, nodes_csv: Path | None, edges_csv: Path | None) -> pd.DataFrame | None:
    """Ilu rodzicow danego typu ma kazdy endpoint - kontekst do interpretacji
    bram. Bez tego nie wiadomo, czy niska brama znaczy 'nieprzydatne', czy
    'brak rodzicow tego typu'."""
    if nodes_csv is None or edges_csv is None:
        return None
    import networkx as nx
    n = pd.read_csv(nodes_csv); e = pd.read_csv(edges_csv)
    t = dict(zip(n["node"], n["node_type"]))
    G = nx.DiGraph(); G.add_edges_from(zip(e["source"], e["target"]))
    cov = {}
    for ep in mat.index:
        if ep not in G:
            continue
        c = {}
        for p in G.predecessors(ep):
            c[t.get(p, "?")] = c.get(t.get(p, "?"), 0) + 1
        cov[ep] = c
    return pd.DataFrame(cov).T.reindex(columns=mat.columns).fillna(0).astype(int)


def draw(mat: pd.DataFrame, out: Path, title: str) -> None:
    lim = max(float(np.abs(mat.values).max()), 1e-6)
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(mat) + 1.8))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([TYPE_LABEL.get(c, c) for c in mat.columns], fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([i.replace("_", " ") for i in mat.index], fontsize=8)
    # wartosci w komorkach - mapa cieplna sama nie pozwala odczytac skali
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 * lim else "black")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def to_latex(mat: pd.DataFrame, cov: pd.DataFrame | None) -> str:
    head = " & ".join(f"{{{c.replace('_', ' ')}}}" for c in mat.columns)
    lines = [
        r"\begin{tabular}{l" + "S" * mat.shape[1] + "}",
        r"  \toprule",
        f"  Endpoint & {head} \\\\",
        r"  \midrule",
    ]
    for ep, row in mat.iterrows():
        cells = []
        for c in mat.columns:
            v = f"{row[c]:.2f}"
            if cov is not None:
                v += f"~({cov.loc[ep, c]})" if ep in cov.index else ""
            cells.append(v)
        lines.append(f"  {ep.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    lines += [r"  \midrule",
              "  Mean & " + " & ".join(f"{mat[c].mean():.2f}" for c in mat.columns) + r" \\",
              r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--run-filter", default=None, help="np. _L3_ albo _L6_")
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--out", default="rysunki")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mat, run = extract(Path(args.runs), args.run_filter)
    order = ["patient_context", "drug_exposure", "mechanism", "adr_or_intermediate_state"]
    mat = mat.reindex(columns=[c for c in order if c in mat.columns])
    mat = mat.loc[mat.abs().sum(axis=1).sort_values(ascending=False).index]

    cov = add_coverage(mat, Path(args.nodes) if args.nodes else None,
                       Path(args.edges) if args.edges else None)

    L = re.search(r"_L(\d+)_", run)
    draw(mat, out / "hcr_gate_matrix", f"HCR gate weights, L={L.group(1) if L else '?'}")
    (out / "hcr_gate_matrix.tex").write_text(to_latex(mat, cov), encoding="utf-8")
    mat.to_csv(out / "hcr_gate_matrix.csv")

    print(f"przebieg: {run}\n")
    print(mat.round(3).to_string())
    print("\nsrednia per typ:")
    print(mat.mean().round(3).to_string())
    if cov is not None:
        print("\nendpointow z rodzicem danego typu:")
        print((cov > 0).sum().to_string())
    print(f"\nzapisano do {out.resolve()}")


if __name__ == "__main__":
    main()
