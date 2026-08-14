"""
Kontrola przed treningiem: czy hide_direct_parents ukrywa WARTOSCI, a nie
usuwa wezly z grafu.

DLACZEGO TO JEST KRYTYCZNE. Eksperyment ma sprawdzic, czy glebokosc odzyskuje
przewage, gdy informacja lezy dalej niz jeden hop. Wymaga to, zeby bezposredni
rodzice endpointu ZOSTALI w grafie jako przekazniki, a zniknela tylko ich
obserwowana wartosc. Gdyby zostali usunieci z topologii, endpointy zostalyby
odciete od reszty grafu i przebieg mierzylby nic.

TEST: liczba krawedzi musi byc IDENTYCZNA w obu wariantach, a liczba
niezerowych wartosci ma spasc.

Uruchom z katalogu, w ktorym normalnie odpalasz train_taskB.py:
    python check_hidden_parents.py
    python check_hidden_parents.py --config-dir ../configs --data dataset_proxy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

NODE_TYPES = ["patient_context", "drug_exposure", "mechanism",
              "adr_or_intermediate_state", "clinical_endpoint"]


def load_cfg(config_dir: str, data_group: str, model_group: str, regime: str, hide: bool):
    """Sklada config tak samo jak train_taskB.py, ale bez uruchamiania main().

    initialize_config_dir wymaga sciezki BEZWZGLEDNEJ - stad resolve().
    """
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"data={data_group}", f"model={model_group}"],
        )
    # open_dict: struct mode Hydry blokuje dopisywanie nowych kluczy
    with open_dict(cfg):
        cfg.data.dataset.observability_regime = regime
        cfg.data.dataset.hide_direct_parents = hide
    return cfg


def summarize(topology, samples_df, regime: str, label: str, build_fn):
    graphs = build_fn(samples_df.head(5), topology, regime)
    g = graphs[0]
    n_edges = sum(v.size(1) for v in g.edge_index_dict.values())
    print(f"\n--- {label} ---")
    for nt in NODE_TYPES:
        if nt not in g.node_types:
            continue
        nz = int((g[nt].x[:, 0] != 0).sum())
        print(f"  {nt:28s} niezerowych wartosci: {nz:5d} / {g[nt].num_nodes}")
    print(f"  krawedzi lacznie: {n_edges:,}")
    print(f"  wykluczonych wezlow (bez wartosci): {len(topology['excluded_nodes'])}")
    return n_edges


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--data", default="dataset_syn")
    ap.add_argument("--model", default="TaskB_gnn_node")
    ap.add_argument("--regime", default="full")
    ap.add_argument("--src", default="src", help="katalog dodawany do sys.path")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.src).resolve()))
    from data.PreprocessingTaskB.build_patient_dag_heterodata_regime import (  # type: ignore
        load_shared_hetero_topology, build_patient_hetero_graphs,
    )
    import pandas as pd

    results = {}
    samples_df = None
    for hide, label in [(False, "BEZ ukrywania"), (True, "Z ukrywaniem rodzicow")]:
        cfg = load_cfg(args.config_dir, args.data, args.model, args.regime, hide)
        topology = load_shared_hetero_topology(cfg)
        if samples_df is None:
            root = Path(cfg.data.dataset.root_dir)
            scenario = str(getattr(cfg.data.dataset, "scenario", "clean"))
            path = root / f"synthetic_pharmacotherapy_v3_samples_{scenario}.csv"
            if not path.exists():
                raise SystemExit(f"Nie znaleziono pliku z probkami: {path}")
            samples_df = pd.read_csv(path)
            print(f"Probki: {path.name}  ({len(samples_df):,} wierszy)")
        results[label] = summarize(
            topology, samples_df, args.regime, label, build_patient_hetero_graphs
        )

    a, b = results["BEZ ukrywania"], results["Z ukrywaniem rodzicow"]
    print("\n" + "=" * 62)
    if a == b:
        print(f"OK: liczba krawedzi identyczna ({a:,}).")
        print("Wezly zostaly w grafie i propaguja - ukryte sa tylko wartosci.")
    else:
        print(f"BLAD: krawedzi {a:,} -> {b:,}.")
        print("Ukrywanie USUWA wezly z topologii, a nie tylko ich wartosci.")
        print("Endpointy zostaly odciete - eksperyment mierzylby nic.")
        print("Sprawdz, czy excluded_nodes nie jest uzywane do filtrowania")
        print("nodes_df albo edges (powinno dzialac tylko na liscie kolumn).")


if __name__ == "__main__":
    main()