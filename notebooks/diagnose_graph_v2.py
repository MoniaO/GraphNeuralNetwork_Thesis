#!/usr/bin/env python3
"""
Diagnostyka grafu farmakoterapii (v2.1 / v2.2) POD KATEM problemu oversmoothingu
w predykcji / typowaniu sciezek przyczynowych.

To jest diagnostyka "przed treningiem": operuje na samej specyfikacji grafu
(nodes + edges) oraz opcjonalnie na jednym wygenerowanym scenariuszu (sample CSV).
Nie wymaga torch/PyG. Odpowiada na pytania:

  1. Czy graf jest poprawnym, spojnym DAG-iem nadajacym sie do zadania?
  2. Jak gleboko trzeba propagowac informacje, by pokryc pelne sciezki
     przyczynowe (kontekst -> ... -> endpoint)?  [wymagane pole recepcyjne]
  3. Jak szybko przy tej glebokosci reprezentacje wezlow ulegaja wygladzeniu?
     [Dirichlet energy + MAD vs. liczba hopow]  -> podatnosc na oversmoothing
  4. Czy graf jest homofiliczny czy heterofiliczny? (heterofilia nasila
     oversmoothing dla klasycznego GCN)
  5. Czy dane nadaja sie do link prediction (czy da sie podzielic relacje
     na train/val/test)?
  6. Sygnaly nieprzydatnosci: wezly izolowane, relacje z 1 krawedzia,
     wezly deterministyczne / potencjalny leakage, prevalencja endpointow.

Uzycie (tylko struktura):
    python diagnose_graph_v2.py \
        --nodes synthetic_pharmacotherapy_v2_nodes.csv \
        --edges synthetic_pharmacotherapy_v2_1_edges_audited.csv \
        --out_dir ./diag_out

Uzycie (struktura + jeden scenariusz):
    python diagnose_graph_v2.py \
        --nodes .../v2_2_nodes.csv --edges .../v2_2_edges_audited.csv \
        --sample .../samples_clean.csv --out_dir ./diag_out
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# matplotlib jest opcjonalny (wykresy). Skrypt dziala i bez niego.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


ENDPOINTS = [
    "AKI", "DILI", "Depression", "Falls", "Delirium", "GI_bleeding",
    "Hyponatremia", "Hyperkalemia", "QT_arrhythmia", "Hospitalization",
]

# Wezly deterministyczne w v2.2 (funkcje swoich rodzicow) -> ryzyko leakage /
# trywialnej predykcji krawedzi. Jesli ktoregos nie ma w grafie, jest pomijany.
DETERMINISTIC_NODES = [
    "active_drug_count", "ddi_renal_double_hit", "ddi_renal_triple_whammy",
    "ddi_cns_depression_synergy", "ddi_bleeding_dual", "ddi_bleeding_triple",
    "ddi_serotonergic_synergy", "ddi_qt_multidrug_load", "ddi_hepatic_triple_hit",
    "cumulative_adr_burden", "severe_adr_burden",
    "elevated_creatinine_recorded", "elevated_alt_ast_recorded", "qt_recorded",
]


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def layer_num(layer_label: object) -> float:
    try:
        return int(str(layer_label).split("_")[0])
    except Exception:
        return np.nan


# --------------------------------------------------------------------------- #
#  Wczytanie grafu
# --------------------------------------------------------------------------- #
def load_graph(nodes_csv: Path, edges_csv: Path, drop_latent: bool = True):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    latent = set(nodes.loc[_truthy(nodes.get("is_latent", pd.Series(dtype=str))), "node"])
    if drop_latent and latent:
        nodes_obs = nodes[~nodes["node"].isin(latent)].reset_index(drop=True)
        edges_obs = edges[~edges["source"].isin(latent) & ~edges["target"].isin(latent)].reset_index(drop=True)
    else:
        nodes_obs, edges_obs = nodes, edges
    G = nx.from_pandas_edgelist(edges_obs, "source", "target", create_using=nx.DiGraph)
    G.add_nodes_from(nodes_obs["node"])
    return G, nodes, nodes_obs, edges, edges_obs, latent


# --------------------------------------------------------------------------- #
#  1-2. Struktura + wymagane pole recepcyjne
# --------------------------------------------------------------------------- #
def structural_report(G, nodes_obs, edges_obs, latent):
    ntype = dict(zip(nodes_obs["node"], nodes_obs["node_type"]))
    layer = dict(zip(nodes_obs["node"], nodes_obs.get("layer", pd.Series(dtype=str))))
    rep = {}
    rep["n_nodes_observed"] = G.number_of_nodes()
    rep["n_edges_observed"] = G.number_of_edges()
    rep["n_latent_dropped"] = len(latent)
    rep["is_dag"] = bool(nx.is_directed_acyclic_graph(G))
    rep["weakly_connected_components"] = nx.number_weakly_connected_components(G)
    isolated = [x for x in G.nodes if G.degree(x) == 0]
    rep["isolated_nodes"] = isolated

    if rep["is_dag"]:
        lp = nx.dag_longest_path(G)
        rep["longest_path_edges"] = len(lp) - 1
        rep["longest_path"] = lp

    # rozklad "skoku" miedzy warstwami
    e = edges_obs.copy()
    e["sl"] = e["source"].map(layer).map(layer_num)
    e["tl"] = e["target"].map(layer).map(layer_num)
    jump = (e["tl"] - e["sl"]).dropna()
    rep["layer_jump_distribution"] = {int(k): int(v) for k, v in sorted(Counter(jump.astype(int)).items())}
    rep["within_or_backward_edges"] = int((jump <= 0).sum())

    # homofilia
    st, tt = e["source"].map(ntype), e["target"].map(ntype)
    rep["edge_homophily_node_type"] = round(float((st == tt).mean()), 3)
    sl_, tl_ = e["source"].map(layer), e["target"].map(layer)
    rep["edge_homophily_layer"] = round(float((sl_ == tl_).mean()), 3)

    # WYMAGANE POLE RECEPCYJNE: kontekst(warstwa 1) -> endpointy (skierowane)
    ctx = [x for x in G.nodes if layer_num(layer.get(x, "")) == 1]
    endp = nodes_obs.loc[_truthy(nodes_obs.get("is_endpoint", pd.Series(dtype=str))), "node"].tolist()
    dists = []
    for c in ctx:
        sp = nx.single_source_shortest_path_length(G, c)
        dists += [sp[t] for t in endp if t in sp]
    if dists:
        d = np.array(dists)
        rep["ctx_to_endpoint_pairs"] = int(len(d))
        rep["ctx_to_endpoint_depth"] = {
            "min": int(d.min()), "median": float(np.median(d)),
            "mean": round(float(d.mean()), 2), "max": int(d.max()),
            "distribution": {int(k): int(v) for k, v in sorted(Counter(d).items())},
        }

    # stopnie / huby
    ind, outd = dict(G.in_degree()), dict(G.out_degree())
    deg = pd.DataFrame({"node": list(G.nodes)})
    deg["in"], deg["out"] = deg["node"].map(ind), deg["node"].map(outd)
    deg["type"] = deg["node"].map(ntype)
    rep["mean_in_degree"] = round(float(deg["in"].mean()), 2)
    rep["top_out_degree_hubs"] = deg.sort_values("out", ascending=False).head(6)[["node", "in", "out", "type"]].to_dict("records")
    rep["top_in_degree_sinks"] = deg.sort_values("in", ascending=False).head(6)[["node", "in", "out", "type"]].to_dict("records")
    return rep, deg


# --------------------------------------------------------------------------- #
#  3. Podatnosc na oversmoothing: Dirichlet energy + MAD vs. glebokosc
# --------------------------------------------------------------------------- #
def oversmoothing_propensity(G, k_max=12, n_feat=64, seed=0):
    """Propagujemy losowe cechy przez symetrycznie znormalizowana macierz
    sasiedztwa z self-loopami (styl GCN) i mierzymy:
      - Dirichlet energy: tr(F^T L F)/||F||^2  (0 => pelne wygladzenie),
      - MAD: srednia odleglosc kosinusowa miedzy para wezlow (0 => kolaps).
    Szybki spadek = wysoka podatnosc na oversmoothing na tej glebokosci."""
    rng = np.random.default_rng(seed)
    U = G.to_undirected()
    nodes = list(G.nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N))
    for u, v in U.edges():
        A[idx[u], idx[v]] = 1.0
        A[idx[v], idx[u]] = 1.0
    A_aug = A + np.eye(N)
    d = A_aug.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(d))
    S = Dinv @ A_aug @ Dinv
    L = np.diag(A_aug.sum(1)) - A_aug

    def dirichlet(F):
        return float(np.trace(F.T @ L @ F) / (np.linalg.norm(F) ** 2 + 1e-12))

    def mad(F):
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        cos = Fn @ Fn.T
        dist = 1.0 - cos
        return float(dist[~np.eye(N, dtype=bool)].mean())

    F = rng.standard_normal((N, n_feat))
    rows = []
    for k in range(k_max + 1):
        rows.append({"k": k, "dirichlet_energy": round(dirichlet(F), 5), "mad": round(mad(F), 5)})
        F = S @ F
    df = pd.DataFrame(rows)

    # algebraiczna spojnosc = lambda_2 znormalizowanego Laplasjanu (bez self-loop)
    dd = A.sum(1).copy()
    dd[dd == 0] = 1.0
    Ln = np.eye(N) - np.diag(1.0 / np.sqrt(dd)) @ A @ np.diag(1.0 / np.sqrt(dd))
    ev = np.sort(np.linalg.eigvalsh(Ln))
    algebraic_conn = float(ev[1]) if N > 1 else float("nan")
    return df, algebraic_conn


# --------------------------------------------------------------------------- #
#  5. Podzielnosc relacji dla link prediction
# --------------------------------------------------------------------------- #
def relation_splittability(edges_obs, nodes_obs):
    ntype = dict(zip(nodes_obs["node"], nodes_obs["node_type"]))
    e = edges_obs.copy()
    e["rel"] = list(zip(e["source"].map(ntype), e["edge_type"], e["target"].map(ntype)))
    rc = e["rel"].value_counts()
    return {
        "n_relation_types": int(rc.shape[0]),
        "relations_with_1_edge": int((rc == 1).sum()),
        "relations_with_le3_edges": int((rc <= 3).sum()),
        "largest_relations": [{"relation": str(k), "n_edges": int(v)} for k, v in rc.head(8).items()],
        "note": "Relacje z <=3 krawedziami nie daja sie sensownie podzielic na train/val/test per-relacja.",
    }


# --------------------------------------------------------------------------- #
#  6. Diagnostyka tabelaryczna (jesli podano sample CSV)
# --------------------------------------------------------------------------- #
def tabular_report(sample_csv: Path, nodes_all: pd.DataFrame):
    df = pd.read_csv(sample_csv)
    rep = {"n_rows": int(len(df)), "n_cols": int(df.shape[1])}

    # prevalencja endpointow (rzadkosc = motywacja metodologiczna, ale i wyzwanie)
    prev = {e: round(float(df[e].mean()), 4) for e in ENDPOINTS if e in df.columns}
    rep["endpoint_prevalence"] = prev
    rep["endpoints_below_3pct"] = [k for k, v in prev.items() if v < 0.03]

    # wezly deterministyczne: sprawdz, czy faktycznie deterministyczne wzgl. rodzicow
    det_present = [c for c in DETERMINISTIC_NODES if c in df.columns]
    rep["deterministic_nodes_present"] = det_present

    # leakage recorded_/true_
    rec = [c for c in df.columns if c.startswith("recorded_")]
    tru = [c for c in df.columns if c.startswith("true_")]
    rep["noisy_doc_columns"] = {"recorded_": len(rec), "true_": len(tru)}
    if rec and tru:
        rep["noisy_doc_warning"] = ("Kolumny true_* to sygnal biologiczny (leakage jesli "
                                    "trafia do cech). Do uczenia uzywac recorded_*.")

    # zerowa wariancja / stale kolumny (martwe cechy)
    numcols = df.select_dtypes(include=[np.number]).columns
    const = [c for c in numcols if df[c].nunique(dropna=True) <= 1]
    rep["constant_columns"] = const[:30]
    rep["n_constant_columns"] = len(const)
    return rep, df


# --------------------------------------------------------------------------- #
#  Wykresy
# --------------------------------------------------------------------------- #
def make_plots(smooth_df, deg, out_dir: Path, tag: str):
    if not HAVE_MPL:
        return []
    paths = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(smooth_df["k"], smooth_df["mad"], "o-", label="MAD")
    ax[0].plot(smooth_df["k"], smooth_df["dirichlet_energy"] / smooth_df["dirichlet_energy"].iloc[0],
               "s--", label="Dirichlet (znorm.)")
    ax[0].set_xlabel("liczba hopow / warstw k"); ax[0].set_ylabel("wartosc")
    ax[0].set_title("Podatnosc na oversmoothing"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].hist(deg["out"], bins=range(0, int(deg["out"].max()) + 2), alpha=.7, label="out")
    ax[1].hist(deg["in"], bins=range(0, int(deg["in"].max()) + 2), alpha=.7, label="in")
    ax[1].set_xlabel("stopien"); ax[1].set_ylabel("liczba wezlow")
    ax[1].set_title("Rozklad stopni"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    p = out_dir / f"diag_{tag}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    paths.append(str(p))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--edges", required=True, type=Path)
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=Path("./diag_out"))
    ap.add_argument("--k_max", type=int, default=12)
    ap.add_argument("--keep_latent", action="store_true",
                    help="Nie usuwaj wezlow latentnych (domyslnie usuwane, zgodnie z PLAN.md).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    G, nodes_all, nodes_obs, edges_all, edges_obs, latent = load_graph(
        args.nodes, args.edges, drop_latent=not args.keep_latent)

    report = {"inputs": {"nodes": str(args.nodes), "edges": str(args.edges),
                         "dropped_latent": sorted(latent) if not args.keep_latent else []}}
    struct, deg = structural_report(G, nodes_obs, edges_obs, latent)
    report["structure"] = struct
    smooth_df, alg_conn = oversmoothing_propensity(G, k_max=args.k_max)
    report["oversmoothing"] = {
        "algebraic_connectivity": round(alg_conn, 5),
        "mad_at_required_depth": None,
        "curve": smooth_df.to_dict("records"),
    }
    # MAD na medianie glebokosci sciezek kontekst->endpoint
    med = struct.get("ctx_to_endpoint_depth", {}).get("median")
    if med is not None:
        kk = int(round(med))
        row = smooth_df.loc[smooth_df["k"] == min(kk, args.k_max)]
        if len(row):
            report["oversmoothing"]["mad_at_required_depth"] = {
                "k": kk, "mad": float(row["mad"].iloc[0]),
                "mad_drop_vs_k0": round(1 - float(row["mad"].iloc[0]) / float(smooth_df["mad"].iloc[0]), 3),
            }
    report["link_prediction_splittability"] = relation_splittability(edges_obs, nodes_obs)

    if args.sample is not None and args.sample.exists():
        tab, _ = tabular_report(args.sample, nodes_all)
        report["tabular"] = tab

    tag = args.nodes.stem
    smooth_df.to_csv(args.out_dir / f"oversmoothing_curve_{tag}.csv", index=False)
    plots = make_plots(smooth_df, deg, args.out_dir, tag)
    report["plots"] = plots
    (args.out_dir / f"diagnostics_{tag}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
