from __future__ import annotations

"""HCR (Duda) jako "wide" sciezka Wide&Deep dla Task B - binarne pary
predyktor<->endpoint, wprost wg sekcji 9.1/9.2 dokumentu HCR/GHCR.

Rozniczka wzgledem "bezpiecznej" wersji edge-level (miedzy sasiadami w DAG,
cohort-level, bez etykiet): TUTAJ wspolczynnik a_11^(e) jest liczony z
UDZIALEM etykiety endpointu (g_e(y) uzywa y_ie wprost) - to jest zgodne z
dokumentem, ale wymaga ochrony przed leakage. Rozwiazanie: DOKLADNY
leave-one-out (nie K-fold) - kazdy pacjent w train dostaje wspolczynnik
policzony ze WSZYSTKICH POZOSTALYCH pacjentow, nigdy z wlasnej etykiety.
Zamkniety wzor (nie petla po foldach), bo a_11 jest zwykla srednia:

    a_(-i) = (N*a_full - iloczyn_i) / (N-1)

Zakres: TYLKO bezposredni rodzice kazdego endpointu w audytowanym DAG-u
(nie wszystkie 137 binarnych wezlow x 10 endpointow) - zachowuje
interpretacje przyczynowa, unika kombinatorycznej eksplozji (43 pary
zamiast 1370). Wezly wykluczone jako leakage (unobserved_severity,
hospital_contact, ...) sa pomijane tak samo jak w reszcie pipeline'u.

Wynik: macierz s_p,e (pacjent x endpoint), gotowa do dolaczenia do grafu
jako "wide" skladnik dodawany do logitu obok "deep" sciezki GNN - patrz
build_patient_dag_heterodata.py (dolaczenie) i gnn_node.py (dodanie
do logitu w forward()).
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def get_binary_direct_parents(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    target_endpoints: List[str],
    excluded_nodes: set,
) -> Dict[str, List[str]]:
    """Bezposredni rodzice kazdego endpointu w DAG-u, ograniczeni do
    wezlow o wywnioskowanym value_type=='binary' i niebedacych w
    excluded_nodes (leakage: unobserved_severity, hospital_contact, ...).
    """
    value_type = infer_value_types(nodes)
    parents_by_endpoint: Dict[str, List[str]] = {}
    for ep in target_endpoints:
        parents = edges.loc[edges["target"] == ep, "source"].tolist()
        binary_parents = [
            p for p in parents
            if p not in excluded_nodes and value_type.get(p) == "binary"
        ]
        parents_by_endpoint[ep] = binary_parents
    return parents_by_endpoint


def infer_value_types(nodes: pd.DataFrame) -> Dict[str, str]:
    """value_type z metadanych, z fallbackiem dla wezlow gdzie kolumna jest
    pusta (patrz analiza: 106/155 wezlow nie ma zadeklarowanego value_type -
    w tym WSZYSTKIE zmienne ciagle z patient_context, np. age/baseline_egfr).
    Fallback bazuje na znanej liscie zmiennych ciaglych/count z patient_context;
    wszystko inne bez deklaracji traktowane jako binarne (co odpowiada
    faktycznej strukturze reszty grafu - patrz weryfikacja w rozmowie)."""
    continuous_pc = {
        "age", "baseline_alt_ast", "baseline_egfr", "baseline_potassium",
        "baseline_sodium", "monitoring_intensity",
    }
    count_pc = {"polypharmacy_burden"}

    declared = dict(zip(nodes["node"], nodes["value_type"]))
    result: Dict[str, str] = {}
    for node in nodes["node"]:
        d = declared.get(node)
        if isinstance(d, str) and d in ("count", "continuous"):
            result[node] = d
        elif node in continuous_pc:
            result[node] = "continuous"
        elif node in count_pc:
            result[node] = "count"
        else:
            result[node] = "binary"
    return result


def _phi1_binary(values: np.ndarray, p: float) -> np.ndarray:
    """Standaryzowany kontrast binarny: (x-p)/sqrt(p(1-p))."""
    denom = np.sqrt(max(p * (1.0 - p), 1e-12))
    return (values - p) / denom


def compute_hcr_wide_scores(
    samples_df: pd.DataFrame,
    parents_by_endpoint: Dict[str, List[str]],
    train_patient_ids: set,
) -> pd.DataFrame:
    """Zwraca DataFrame indeksowany patient_id, kolumny = endpointy z
    parents_by_endpoint, wartosci = s_p,e = suma po rodzicach
    a_r^(e) * phi_1(x_pr) (agregacja: prosta suma, patrz sekcja 9.2 -
    dokument nie narzuca konkretnego agregatora poza "Aggregator").

    Wspolczynniki a_r^(e):
    - dla pacjentow TRENINGOWYCH: dokladny leave-one-out (bez wlasnej etykiety)
    - dla pacjentow VALID/TEST: pelny wspolczynnik z CALEGO train (bezpieczne -
      te zbiory nigdy nie wspoluczestnicza w dopasowaniu wspolczynnikow)

    Standaryzacja p_r (predyktora) liczona raz na train (jak reszta
    pipeline'u, np. compute_norm_stats) - NIE wymaga cross-fittingu, bo nie
    uzywa etykiety endpointu, wiec nie ma tam ryzyka leakage etykiety.
    """
    if "patient_id" not in samples_df.columns:
        raise ValueError("samples_df musi zawierac kolumne 'patient_id'.")

    train_mask = samples_df["patient_id"].isin(train_patient_ids)
    train_df = samples_df.loc[train_mask]
    n_train = len(train_df)
    if n_train < 2:
        raise ValueError("Za malo pacjentow treningowych do policzenia HCR (potrzeba >= 2).")

    out = pd.DataFrame({"patient_id": samples_df["patient_id"].to_numpy()})

    for endpoint, parents in parents_by_endpoint.items():
        if endpoint not in samples_df.columns:
            raise ValueError(f"Brak kolumny endpointu '{endpoint}' w samples_df.")

        y_train = train_df[endpoint].to_numpy(dtype=float)
        p_e_full = float(y_train.mean())
        N = n_train

        score_full = np.zeros(len(samples_df), dtype=float)  # dla WSZYSTKICH pacjentow (train+valid+test)
        # osobno budujemy wektor "wynik dla kazdego train-pacjenta" (LOO),
        # zeby na koncu poprawnie wstawic go z powrotem pod ich patient_id
        score_train_loo = np.zeros(n_train, dtype=float)

        if not parents:
            out[endpoint] = 0.0
            continue

        for r in parents:
            if r not in samples_df.columns:
                raise ValueError(f"Brak kolumny rodzica '{r}' w samples_df (endpoint={endpoint}).")

            x_train = train_df[r].to_numpy(dtype=float)
            p_r_full = float(x_train.mean())

            # --- pelny, SCISLE POPRAWNY zamkniety wzor leave-one-out ---
            # W pierwszej wersji tego modulu standaryzacja (p_r, p_e) byla
            # liczona z CALEGO train i tylko licznik korelacji byl LOO - to
            # zostawialo slaby (rzedu 1/N), ale realny przeciek etykiety przez
            # p_e. Ponizej p_r, p_e SA REFITOWANE per pacjent w zamknietej
            # formie (bez petli), wiec wynik jest identyczny z brute-force
            # LOO (zweryfikowane numerycznie do bledu zaokraglenia float64).
            #
            # Wyprowadzenie: dla kowariancji z wylaczonym i-tym pacjentem
            #   cov_(-i) = (n11 - x_i*y_i)/(N-1) - p_r_(-i)*p_e_(-i)
            # gdzie n11 = suma x*y na CALYM train (n_11 = liczba wspolwystapien),
            # a p_r_(-i), p_e_(-i) to srednie z pominieciem pacjenta i (rowniez
            # zamkniete, bez petli).
            n11 = float((x_train * y_train).sum())
            p_r_loo = (N * p_r_full - x_train) / (N - 1)
            p_e_loo = (N * p_e_full - y_train) / (N - 1)
            cov_loo = (n11 - x_train * y_train) / (N - 1) - p_r_loo * p_e_loo
            denom_loo = np.sqrt(
                np.clip(p_r_loo * (1 - p_r_loo) * p_e_loo * (1 - p_e_loo), 1e-12, None)
            )
            a_loo = cov_loo / denom_loo

            # Pelny wspolczynnik (na calym train, bez wylaczania) - uzywany
            # WYLACZNIE dla pacjentow valid/test, ktorzy nigdy nie
            # wspoluczestniczyli w dopasowaniu (bezpieczne bez LOO).
            phi_x_train_full = _phi1_binary(x_train, p_r_full)
            g_e_train = _phi1_binary(y_train, p_e_full)
            a_full = float((phi_x_train_full * g_e_train).mean())

            x_all = samples_df[r].to_numpy(dtype=float)
            phi_x_all = _phi1_binary(x_all, p_r_full)
            score_full += a_full * phi_x_all

            # WAZNE: standaryzacja WLASNEJ wartosci pacjenta w wersji LOO
            # rowniez musi uzywac p_r_loo (nie p_r_full) - inaczej wynik nie
            # jest prawdziwym LOO, mimo ze a_loo samo w sobie jest poprawne
            # (zlapane i poprawione podczas weryfikacji numerycznej wzgledem
            # brute-force: bez tego wystepowala roznica ~0.001 na pacjenta).
            # UWAGA: _phi1_binary zaklada SKALARNE p (max() na tablicy dzialaby
            # bledne - redukcja zamiast elementwise) - p_r_loo jest tablica
            # (osobna wartosc per pacjent), wiec liczymy wprost, nie przez
            # _phi1_binary.
            denom_loo_x = np.sqrt(np.clip(p_r_loo * (1 - p_r_loo), 1e-12, None))
            phi_x_train_loo = (x_train - p_r_loo) / denom_loo_x
            score_train_loo += a_loo * phi_x_train_loo

        # scal: score_full juz ma poprawne wartosci dla valid/test (pelny a_full);
        # dla train nadpisujemy wersja LOO
        train_idx_positions = np.flatnonzero(train_mask.to_numpy())
        score_full[train_idx_positions] = score_train_loo

        out[endpoint] = score_full

    return out
