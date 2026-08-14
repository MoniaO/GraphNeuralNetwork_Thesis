"""
Wczytanie Hetionet v1.0 z LOKALNYCH plikow (pobranych recznie z GitHub).

Zrodlo: https://github.com/hetio/hetionet  (katalog hetnet/tsv)
Publikacja: Himmelstein i in., eLife 2017, DOI: 10.7554/eLife.26726

FORMATY - o co chodzi z ".sif":
  SIF (Simple Interaction Format) to format z Cytoscape, ale w Hetionecie
  jest to zwykly plik tekstowy rozdzielany tabulatorami, z naglowkiem:
      source <TAB> metaedge <TAB> target
  czyli po prostu TSV o innym rozszerzeniu. pandas czyta go bez zadnych
  sztuczek - wystarczy sep="\t". Repo publikuje ten sam plik w wersji
  spakowanej (.sif.gz) i nie (.sif); ten skrypt obsluguje obie.

  Wezly (hetionet-v1.0-nodes.tsv):
      id <TAB> name <TAB> kind
      gdzie id ma postac "Compound::DB00014", "Gene::5743", "Disease::DOID:1234"

Ten skrypt NIC nie pobiera - zaklada, ze pliki juz masz na dysku.
Domyslnie szuka ich w ./data/hetionet, ale przyjmuje dowolna sciezke.

Uzycie:
    python load_hetionet.py                      # szuka w ./data/hetionet
    python load_hetionet.py ~/Downloads          # szuka we wskazanym katalogu

    # albo jako import:
    from load_hetionet import load_hetionet
    nodes, edges = load_hetionet("./data/hetionet")
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DIR = Path("/Users/monika/hetionet")

# Oficjalne statystyki v1.0 - sluza do walidacji, czy plik jest kompletny.
EXPECTED_N_NODES = 47_031
EXPECTED_N_EDGES = 2_250_197
EXPECTED_N_NODE_KINDS = 11
EXPECTED_N_METAEDGES = 24

# Wzorce nazw, ktore moga sie pojawic w zaleznosci od sposobu pobrania.
NODE_PATTERNS = [
    "hetionet-v1.0-nodes.tsv",
    "hetionet-v1.0-nodes.tsv.gz",
    "*nodes*.tsv",
    "*nodes*.tsv.gz",
]
EDGE_PATTERNS = [
    "hetionet-v1.0-edges.sif",
    "hetionet-v1.0-edges.sif.gz",
    "*edges*.sif",
    "*edges*.sif.gz",
    "*edges*.tsv",
    "*edges*.tsv.gz",
]

LFS_POINTER_MAX_BYTES = 1024


def _find_file(directory: Path, patterns: list[str], what: str) -> Path:
    """Znajduje pierwszy pasujacy plik. Wzorce sa uporzadkowane od
    najbardziej do najmniej konkretnego, zeby przy kilku kandydatach
    wybrac ten o oficjalnej nazwie."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Katalog nie istnieje: {directory.resolve()}\n"
            f"Wskaz katalog, w ktorym lezy pobrany Hetionet, np.:\n"
            f"    python load_hetionet.py ~/Downloads"
        )
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    listing = "\n  ".join(sorted(p.name for p in directory.iterdir())) or "(pusty)"
    raise FileNotFoundError(
        f"Nie znalazlam pliku z {what} w {directory.resolve()}\n"
        f"Szukalam wzorcow: {patterns}\n"
        f"Zawartosc katalogu:\n  {listing}"
    )


def _check_not_lfs_pointer(path: Path) -> None:
    """Git LFS bez zainstalowanego git-lfs podstawia ~130-200-bajtowy plik
    tekstowy zamiast prawdziwych danych. Wykrywamy to po rozmiarze i po
    charakterystycznym naglowku, zeby nie dostac pozniej mylacego bledu
    parsowania."""
    size = path.stat().st_size
    if size >= LFS_POINTER_MAX_BYTES:
        return
    head = path.read_bytes()[:300].decode("utf-8", errors="replace")
    if "git-lfs" in head or "oid sha256" in head:
        raise RuntimeError(
            f"{path.name} to wskaznik Git LFS ({size} B), nie prawdziwe dane.\n"
            f"Podglad:\n{head}\n\n"
            "Rozwiazanie: pobierz plik przyciskiem 'Download raw file' na "
            "stronie pliku w GitHub, albo `git lfs install && git lfs pull`."
        )
    # Maly plik BEZ markerow LFS to nie blad - moze byc swiadomie
    # przygotowany podzbior do testow. Ostrzegamy i idziemy dalej.
    print(f"  UWAGA: {path.name} jest maly ({size} B) - upewnij sie, ze to celowe.")


def _read_table(path: Path) -> pd.DataFrame:
    """Czyta TSV, przezroczyscie obslugujac .gz. Rozszerzenie (.sif/.tsv)
    nie ma znaczenia - liczy sie separator."""
    _check_not_lfs_pointer(path)
    print(f"  czytam: {path.name} ({path.stat().st_size / 1e6:,.1f} MB)")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


def _validate_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: brakuje kolumn {missing}. Znalezione kolumny: {list(df.columns)}\n"
            "Jesli kolumny wygladaja jak dane (np. 'Compound::DB00014'), to plik "
            "nie ma naglowka - dodaj header=None i names=... w _read_table()."
        )


def load_hetionet(
    data_dir: Path | str = DEFAULT_DIR,
    nodes_path: Optional[Path | str] = None,
    edges_path: Optional[Path | str] = None,
    add_kind_columns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zwraca (nodes, edges).

    nodes: id | name | kind                    [+ nic wiecej]
    edges: source | metaedge | target          [+ source_kind, target_kind
                                                jesli add_kind_columns=True]

    add_kind_columns rozbija prefiks z id ("Compound::DB00014" -> "Compound")
    i dokleja typ zrodla/celu do krawedzi. To jest wygodne przy filtrowaniu
    metaedge'y (np. wybraniu Compound-binds-Gene) bez ciaglego stringowania.
    """
    data_dir = Path(data_dir)

    print("=== Wczytywanie Hetionet v1.0 ===")
    n_path = Path(nodes_path) if nodes_path else _find_file(data_dir, NODE_PATTERNS, "wezlami")
    e_path = Path(edges_path) if edges_path else _find_file(data_dir, EDGE_PATTERNS, "krawedziami")

    nodes = _read_table(n_path)
    _validate_columns(nodes, ["id", "name", "kind"], "nodes")

    edges = _read_table(e_path)
    _validate_columns(edges, ["source", "metaedge", "target"], "edges")

    if add_kind_columns:
        # Prefiks przed "::" to typ wezla - stabilny w calym v1.0.
        edges["source_kind"] = edges["source"].str.split("::", n=1).str[0]
        edges["target_kind"] = edges["target"].str.split("::", n=1).str[0]

    return nodes, edges


def summarize(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Walidacja tresci wzgledem oficjalnych statystyk v1.0. Rozbieznosc
    NIE przerywa dzialania - tylko ostrzega, bo mozesz swiadomie pracowac
    na podzbiorze."""
    print("\n=== Wezly ===")
    ok_n = len(nodes) == EXPECTED_N_NODES
    print(f"Liczba wezlow: {len(nodes):,}  (oczekiwane {EXPECTED_N_NODES:,})  {'OK' if ok_n else 'UWAGA'}")
    kinds = nodes["kind"].value_counts()
    ok_k = len(kinds) == EXPECTED_N_NODE_KINDS
    print(f"Typow wezlow: {len(kinds)}  (oczekiwane {EXPECTED_N_NODE_KINDS})  {'OK' if ok_k else 'UWAGA'}")
    print(kinds.to_string())

    print("\n=== Krawedzie ===")
    ok_e = len(edges) == EXPECTED_N_EDGES
    print(f"Liczba krawedzi: {len(edges):,}  (oczekiwane {EXPECTED_N_EDGES:,})  {'OK' if ok_e else 'UWAGA'}")
    ok_m = edges["metaedge"].nunique() == EXPECTED_N_METAEDGES
    print(f"Typow relacji: {edges['metaedge'].nunique()}  (oczekiwane {EXPECTED_N_METAEDGES})  {'OK' if ok_m else 'UWAGA'}")

    print("\nWszystkie metaedge'y (skrot -> liczba krawedzi):")
    print(edges["metaedge"].value_counts().to_string())

    if "source_kind" in edges.columns:
        print("\n=== Relacje istotne dla planowanego ciecia (Compound jako instancja) ===")
        interesting = edges[
            (edges["source_kind"] == "Compound") | (edges["target_kind"] == "Compound")
        ]
        summary = (
            interesting.groupby(["metaedge", "source_kind", "target_kind"])
            .size()
            .rename("n_edges")
            .reset_index()
            .sort_values("n_edges", ascending=False)
        )
        print(summary.to_string(index=False))

        # Referencyjne liczby z publikacji - szybki test, czy ciecie ma sens.
        n_compounds = int((nodes["kind"] == "Compound").sum())
        cbg = int((edges["metaedge"] == "CbG").sum())
        if n_compounds and cbg:
            print(
                f"\nSrednio genow (CbG) na zwiazek: {cbg / n_compounds:.1f}"
                f"   [dla porownania: 4.7 bezposrednich rodzicow na endpoint w syntetycznym DAG-u]"
            )


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    nodes_df, edges_df = load_hetionet(directory)
    summarize(nodes_df, edges_df)