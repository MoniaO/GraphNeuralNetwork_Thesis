#!/usr/bin/env python3
"""Macierz bram HCR (endpoint x blok dowodowy) z eksportu W&B.

CO TO POKAZUJE. W trybie typed_concat kazdy blok dowodowy jest mnozony przez
uczony skalar hcr_type_gate[endpoint, blok]. Brama startuje od 1.0 i model sam
decyduje, na ile ufa dowodowi z danego bloku przy danym endpoincie. Wartosc
ponizej jedynki oznacza REDUKCJE zaufania, nie wzmocnienie.

Blokow jest cztery (typy rodzicow) albo piec, jesli wlaczono momenty trojne
(hcr_use_triples=true) - wtedy piata kolumna "triple" odpowiada dowodowi
z trojek dziadek-rodzic-endpoint.

DWA RODZAJE PUSTEJ KOMORKI - kluczowe dla interpretacji:
  - endpoint NIE MA rodzica danego typu -> komorka pusta ("n/a"). Maskowana
    srednia zwraca wektor zerowy, gradient nie dochodzi do bramy i zostaje
    ona na wartosci poczatkowej. To nie jest decyzja modelu.
  - endpoint MA rodzica, a brama zeszla do zera -> komorka pokazuje 0.00.
    To jest wyuczone wyciszenie kanalu.
Kolumna "triple" nie jest typem wezla, wiec pokrycia sie dla niej nie liczy
i nigdy nie jest maskowana.

Uzycie:
    python gate_matrix.py --runs plik.csv --run-filter syn_clean_full_nores_ep200_L3 \\
        --nodes nodes.csv --edges edges.csv --out rysunki/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Kolejnosc kolumn na rysunku: warstwy DAG-u od kontekstu do stanu posredniego,
# na koncu blok trojek (jesli obecny).
COLUMN_ORDER = [
    "patient_context",
    "drug_exposure",
    "mechanism",
    "adr_or_intermediate_state",
    "triple",
]

TYPE_LABEL = {
    "patient_context": "patient\ncontext",
    "drug_exposure": "drug\nexposure",
    "mechanism": "mechanism",
    "adr_or_intermediate_state": "adr /\nintermediate",
    "triple": "triple\n(G-P-E)",
}

# Kolumny, dla ktorych pokrycie po typie wezla nie ma sensu - nie sa typami.
NON_NODE_COLUMNS = {"triple"}


def extract(runs_csv: Path, run_filter: str | None) -> tuple[pd.DataFrame, str]:
    """Wyciaga macierz bram z ostatniej epoki wskazanego przebiegu."""
    d = pd.read_csv(runs_csv, low_memory=False)
    gate_cols = [c for c in d.columns if c.startswith("hcr_gate/") and "abs" not in c]
    if not gate_cols:
        raise SystemExit(
            "Brak kolumn hcr_gate/ - przebieg bez typed_concat albo logowanie wylaczone."
        )

    names = [
        r for r in d.run_name.unique()
        if "typed_concat" in r and (run_filter is None or run_filter in r)
    ]
    if not names:
        raise SystemExit(f"Brak przebiegu typed_concat pasujacego do {run_filter!r}.")
    if len(names) > 1:
        print(f"UWAGA: pasuje {len(names)} przebiegow, biore pierwszy. "
              f"Zawez --run-filter, jesli to nie ten:")
        for n in names:
            print(f"    {n}")
    run = names[0]

    g = d[d.run_name == run].sort_values("_step")
    last = g[gate_cols].dropna(how="all").iloc[-1]

    rows: dict[str, dict[str, float]] = {}
    for c in gate_cols:
        _, ep, block = c.split("/", 2)
        rows.setdefault(ep, {})[block] = float(last[c])
    mat = pd.DataFrame(rows).T

    # Eksport moze zawierac przebiegi z KILKU zbiorow naraz (np. syn i proxy),
    # a kolumny hcr_gate/ sa wspolne dla calego pliku. Endpointy nienalezace
    # do wybranego przebiegu maja wtedy same NaN - odrzucamy je, inaczej
    # trafilyby na rysunek jako puste wiersze.
    n_before = len(mat)
    mat = mat.dropna(how="all")
    if len(mat) < n_before:
        print(f"pominieto {n_before - len(mat)} endpointow spoza tego przebiegu "
              f"(inny zbior w tym samym eksporcie)")
    return mat, run


def add_coverage(
    mat: pd.DataFrame, nodes_csv: Path | None, edges_csv: Path | None
) -> pd.DataFrame | None:
    """Ilu rodzicow danego TYPU ma kazdy endpoint.

    Kontekst do interpretacji bram: bez tego nie wiadomo, czy niska wartosc
    znaczy "kanal nieprzydatny", czy "brak rodzicow tego typu".
    Kolumna "triple" jest pomijana - nie jest typem wezla.
    """
    if nodes_csv is None or edges_csv is None:
        return None
    import networkx as nx

    n = pd.read_csv(nodes_csv)
    e = pd.read_csv(edges_csv)
    type_of = dict(zip(n["node"], n["node_type"]))
    G = nx.DiGraph()
    G.add_edges_from(zip(e["source"], e["target"]))

    cov: dict[str, dict[str, int]] = {}
    for ep in mat.index:
        if ep not in G:
            continue
        counts: dict[str, int] = {}
        for p in G.predecessors(ep):
            t = type_of.get(p, "?")
            counts[t] = counts.get(t, 0) + 1
        cov[ep] = counts

    node_type_cols = [c for c in mat.columns if c not in NON_NODE_COLUMNS]
    return (pd.DataFrame(cov).T
            .reindex(columns=node_type_cols)
            .fillna(0)
            .astype(int))


def _is_missing(ep: str, col: str, cov: pd.DataFrame | None) -> bool:
    """Czy komorka odpowiada brakowi rodzicow (a nie wyuczonemu zeru)."""
    if cov is None or col in NON_NODE_COLUMNS:
        return False
    if col not in cov.columns or ep not in cov.index:
        return False
    return bool(cov.loc[ep, col] == 0)


def draw(mat: pd.DataFrame, cov: pd.DataFrame | None, out: Path, title: str) -> None:
    M = mat.copy()
    for ep in M.index:
        for c in M.columns:
            if _is_missing(ep, c, cov):
                M.loc[ep, c] = np.nan

    lim = max(float(np.nanmax(np.abs(M.values))), 1e-6)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#E8E6E3")            # jasny szary = brak rodzicow

    fig, ax = plt.subplots(figsize=(0.95 * M.shape[1] + 2.2, 0.42 * len(M) + 2.0))
    im = ax.imshow(np.ma.masked_invalid(M.values), cmap=cmap,
                   vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([TYPE_LABEL.get(c, c) for c in M.columns], fontsize=8)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([i.replace("_", " ") for i in M.index], fontsize=8)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=7, color="#8A8580", style="italic")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * lim else "black")

    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cov is not None:
        fig.text(0.5, 0.005, "n/a = endpoint has no parent of this type",
                 ha="center", fontsize=7, color="#8A8580", style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
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
            if _is_missing(ep, c, cov):
                cells.append("{--}")          # brak rodzica: nie mylic z zerem
            else:
                v = f"{row[c]:.2f}"
                if cov is not None and c not in NON_NODE_COLUMNS and ep in cov.index:
                    v += f"~({cov.loc[ep, c]})"
                cells.append(v)
        lines.append(f"  {ep.replace('_', ' ')} & " + " & ".join(cells) + r" \\")

    # Srednia LICZONA TYLKO po komorkach z rodzicami - inaczej zera
    # strukturalne zanizaja ja sztucznie.
    means = []
    for c in mat.columns:
        keep = [ep for ep in mat.index if not _is_missing(ep, c, cov)]
        means.append(f"{mat.loc[keep, c].mean():.2f}" if keep else "{--}")
    lines += [
        r"  \midrule",
        "  Mean$^{*}$ & " + " & ".join(means) + r" \\",
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--run-filter", default=None,
                    help="fragment nazwy przebiegu, np. syn_clean_full_nores_ep200_L3")
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--out", default="rysunki")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mat, run = extract(Path(args.runs), args.run_filter)
    mat = mat.reindex(columns=[c for c in COLUMN_ORDER if c in mat.columns])
    mat = mat.loc[mat.abs().sum(axis=1).sort_values(ascending=False).index]

    cov = add_coverage(
        mat,
        Path(args.nodes) if args.nodes else None,
        Path(args.edges) if args.edges else None,
    )

    L = re.search(r"_L(\d+)_", run)
    draw(mat, cov, out / "hcr_gate_matrix",
         f"HCR gate weights, L={L.group(1) if L else '?'}")
    (out / "hcr_gate_matrix.tex").write_text(to_latex(mat, cov), encoding="utf-8")
    mat.to_csv(out / "hcr_gate_matrix.csv")

    print(f"przebieg: {run}\n")
    print(mat.round(3).to_string())

    print("\nsrednia per blok (tylko komorki z rodzicami):")
    for c in mat.columns:
        keep = [ep for ep in mat.index if not _is_missing(ep, c, cov)]
        n_zero = sum(1 for ep in keep if abs(mat.loc[ep, c]) < 1e-6)
        label = "wszystkie" if c in NON_NODE_COLUMNS else f"{len(keep)} z {len(mat)}"
        print(f"  {c:28s} {mat.loc[keep, c].mean():.3f}   "
              f"(endpointow: {label}, wyuczonych zer: {n_zero})")

    print(f"\nzapisano do {out.resolve()}")


if __name__ == "__main__":
    main()