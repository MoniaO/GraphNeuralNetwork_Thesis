from __future__ import annotations

"""

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

from typing import Dict, List, Optional, Tuple

import hashlib

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

#limit not only to binary relation between parent and endpoint, but also to the parent node type (drug_exposure, mechanism, adr_or_intermediate_state, patient_context)
def get_direct_parents_by_type(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    target_endpoints: List[str],
    excluded_nodes: set,
) -> Dict[str, List[Tuple[str, str]]]:

    value_type = infer_value_types(nodes)
    parents_by_endpoint: Dict[str, List[Tuple[str, str]]] = {}
    for ep in target_endpoints:
        parents = edges.loc[edges["target"] == ep, "source"].tolist()
        typed_parents = [
            (p, value_type.get(p, "binary")) for p in parents if p not in excluded_nodes
        ]
        parents_by_endpoint[ep] = typed_parents
    return parents_by_endpoint


# parent endpoint nodes
PARENT_NODE_TYPES = ["drug_exposure", "mechanism", "adr_or_intermediate_state", "patient_context"]

# calcualtion for node type to encode different encoders per type
def get_direct_parents_by_node_type(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    target_endpoints: List[str],
    excluded_nodes: set,
) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:

    value_type = infer_value_types(nodes)
    node_type_map = dict(zip(nodes["node"], nodes["node_type"]))
    result: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for ep in target_endpoints:
        parents = edges.loc[edges["target"] == ep, "source"].tolist()
        by_type: Dict[str, List[Tuple[str, str]]] = {nt: [] for nt in PARENT_NODE_TYPES}
        for p in parents:
            if p in excluded_nodes:
                continue
            nt = node_type_map.get(p)
            if nt not in PARENT_NODE_TYPES:
                print(f"[get_direct_parents_by_node_type] Pomijam rodzica '{p}' "
                      f"(endpoint={ep}) - nieoczekiwany node_type={nt!r}, "
                      f"spoza PARENT_NODE_TYPES={PARENT_NODE_TYPES}.")
                continue
            by_type[nt].append((p, value_type.get(p, "binary")))
        result[ep] = by_type
    return result

#reliability of edges: mechanistic_confidence * evidence_weights
def get_edge_reliability_weights(
    edges: pd.DataFrame,
    parents_by_endpoint: Dict[str, List],
) -> Dict[str, Dict[str, float]]:

    weights: Dict[str, Dict[str, float]] = {}
    for ep, parents in parents_by_endpoint.items():
        parent_names = [p[0] if isinstance(p, tuple) else p for p in parents]
        ep_weights: Dict[str, float] = {}
        for p in parent_names:
            row = edges[(edges["source"] == p) & (edges["target"] == ep)]
            if len(row) == 0:
                ep_weights[p] = 1.0
                continue
            conf = row["mechanistic_confidence"].iloc[0]
            evid = row["evidence_weight"].iloc[0]
            conf = float(conf) if pd.notna(conf) else 1.0
            evid = float(evid) if pd.notna(evid) else 1.0
            ep_weights[p] = float(np.clip(conf * evid, 1e-3, None))
        weights[ep] = ep_weights
    return weights

#value type based on nodes file 
def infer_value_types(nodes: pd.DataFrame) -> Dict[str, str]:

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

#binary approach
def _phi1_binary(values: np.ndarray, p: float) -> np.ndarray:
    denom = np.sqrt(max(p * (1.0 - p), 1e-12))
    return (values - p) / denom


def _classical4_from_contingency(
    p_r: np.ndarray, p_e: np.ndarray, p11: np.ndarray, eps: float = 1e-6
) -> Dict[str, np.ndarray]:
    """5 statystyk binarnych z tabeli 2x2 (a11 = HCR phi-coefficient,
    NMI, scaled log OR, RD, scaled log-lift - sekcja 3.1/4.2 dokumentu HCR).
    Wektoryzowane - dziala zarowno na skalarach jak i tablicach (LOO per
    pacjent). Zweryfikowane numerycznie wobec brute-force (patrz rozmowa)."""
    p_r = np.clip(p_r, eps, 1 - eps)
    p_e = np.clip(p_e, eps, 1 - eps)
    p11 = np.clip(p11, eps, None)
    p10 = np.clip(p_r - p11, eps, None)
    p01 = np.clip(p_e - p11, eps, None)
    p00 = np.clip(1 - p_r - p_e + p11, eps, None)

    h_r = -p_r * np.log(p_r) - (1 - p_r) * np.log(1 - p_r)
    h_e = -p_e * np.log(p_e) - (1 - p_e) * np.log(1 - p_e)
    i_xy = (
        p11 * np.log(p11 / (p_r * p_e))
        + p10 * np.log(p10 / (p_r * (1 - p_e)))
        + p01 * np.log(p01 / ((1 - p_r) * p_e))
        + p00 * np.log(p00 / ((1 - p_r) * (1 - p_e)))
    )
    nmi = i_xy / np.sqrt(np.clip(h_r * h_e, eps, None))

    odds_ratio = (p11 * p00) / (p10 * p01)
    scaled_log_or = np.tanh(np.log(odds_ratio) / 4.0)

    rd = p11 / p_r - p01 / (1 - p_r)
    scaled_log_lift = np.tanh(np.log(p11 / (p_r * p_e)) / 4.0)

    a11 = (p11 - p_r * p_e) / np.sqrt(p_r * (1 - p_r) * p_e * (1 - p_e))

    return {
        "a11": a11, "nmi": nmi, "scaled_log_or": scaled_log_or,
        "rd": rd, "scaled_log_lift": scaled_log_lift,
    }


# Kolejnosc kanalow w wektorze dowodowym per (pacjent, rodzic, endpoint) -
# uzywana zarowno przy budowie danych, jak i w koderze modelu (musi byc
# spojna po obu stronach).
PAIR_EVIDENCE_CHANNELS = [
    "phi1_patient",   # standaryzowana WLASNA wartosc PACJENTA dla tego rodzica
    "a11", "nmi", "scaled_log_or", "rd", "scaled_log_lift",  # classical4 + a11
    "prevalence_parent", "prevalence_endpoint",              # marginesy (E3)
    "support",                                                 # jaki % train mial oba zmierzone
    "s_evidence"
]
N_PAIR_EVIDENCE_CHANNELS = len(PAIR_EVIDENCE_CHANNELS)

#legendre for continous / count relation
def _legendre_psi(u: np.ndarray, degree: int = 4) -> np.ndarray:
    """[N, degree] macierz przesunietych, znormalizowanych wielomianow
    Legendre'a: psi_k(u) = sqrt(2k+1)*P_k(2u-1), k=1..degree, u w (0,1)
    (sekcja 5 dokumentu HCR Task A)."""
    x = 2.0 * u - 1.0
    P = [np.ones_like(x), x]
    for k in range(2, degree + 1):
        P.append(((2 * k - 1) * x * P[-1] - (k - 1) * P[-2]) / k)
    return np.stack([np.sqrt(2 * k + 1) * P[k] for k in range(1, degree + 1)], axis=1)


def _train_ecdf_transform(x_train: np.ndarray, x_query: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Pseudo-obserwacja u=EDF_train(x) w (0,1), dopasowana WYLACZNIE na
    train, zastosowana do dowolnych wartosci x_query (w tym spoza zakresu
    train - wtedy przycieta do (eps, 1-eps), analogicznie do standardowego
    zachowania EDF poza obserwowanym zakresem)."""
    sorted_train = np.sort(x_train)
    N = len(sorted_train)
    ranks = np.searchsorted(sorted_train, x_query, side="right")
    u = (ranks - 0.5) / N
    return np.clip(u, eps, 1 - eps)

#jitter for continous
def _deterministic_jitter(patient_ids: np.ndarray, var_name: str, seed: int = 0) -> np.ndarray:
    """xi_i w (0,1), deterministyczne z (patient_id, nazwa_zmiennej, seed) -
    powtarzalne miedzy uruchomieniami, bez zaleznosci od globalnego RNG"""
    xi = np.empty(len(patient_ids), dtype=np.float64)
    for i, pid in enumerate(patient_ids):
        h = hashlib.sha256(f"{pid}_{var_name}_{seed}".encode()).hexdigest()
        xi[i] = (int(h[:16], 16) % 10_000_000) / 10_000_000.0
    return xi


def _count_to_uniform(
    x_train: np.ndarray, x_query: np.ndarray, patient_ids_query: np.ndarray,
    var_name: str, seed: int = 0, eps: float = 1e-4,
) -> np.ndarray:
    """Deterministyczny distributional jitter dla count (sekcja 6):
    u = F_C(c^-) + xi*P(C=c), rozklada remisy wewnatrz skoku dystrybuanty
    zamiast jednego wspolnego midranku."""
    values, counts = np.unique(x_train, return_counts=True)
    N = len(x_train)
    cdf_left: Dict[float, float] = {}
    pmf: Dict[float, float] = {}
    cum = 0.0
    for v, c in zip(values, counts):
        cdf_left[v] = cum / N
        pmf[v] = c / N
        cum += c
    max_val = values[-1] if len(values) else 0.0

    def _lookup(v: float) -> Tuple[float, float]:
        if v in cdf_left:
            return cdf_left[v], pmf[v]
        # wartosc spoza train (widziana tylko w valid/test) - bezpieczny,
        # udokumentowany fallback: traktuj jak "powyzej zakresu train"
        if v > max_val:
            return 1.0 - eps, eps
        return eps, eps

    cdf_arr = np.empty(len(x_query))
    pmf_arr = np.empty(len(x_query))
    for i, v in enumerate(x_query):
        cdf_arr[i], pmf_arr[i] = _lookup(v)

    xi = _deterministic_jitter(patient_ids_query, var_name, seed)
    u = cdf_arr + xi * pmf_arr
    return np.clip(u, eps, 1 - eps)


def _pairwise_coefficients_loo(
    y_train: np.ndarray, basis_train: np.ndarray, train_positions: np.ndarray, n_total: int,
) -> Tuple[np.ndarray, np.ndarray]:

    N = len(y_train)
    K = basis_train.shape[1]
    p_e_full = float(y_train.mean())
    denom_full = np.sqrt(max(p_e_full * (1 - p_e_full), 1e-12))

    S_g = basis_train.sum(axis=0)                      # [K]
    S_Yg = (y_train[:, None] * basis_train).sum(axis=0)  # [K]
    a_full = (S_Yg / N - p_e_full * (S_g / N)) / denom_full  # [K]

    p_e_loo = (N * p_e_full - y_train) / (N - 1)                       # [N]
    g_mean_loo = (S_g[None, :] - basis_train) / (N - 1)                # [N,K]
    Yg_mean_loo = (S_Yg[None, :] - y_train[:, None] * basis_train) / (N - 1)  # [N,K]
    cov_loo = Yg_mean_loo - p_e_loo[:, None] * g_mean_loo
    denom_loo = np.sqrt(np.clip(p_e_loo * (1 - p_e_loo), 1e-12, None))
    a_loo = cov_loo / denom_loo[:, None]

    return a_loo, a_full

#nonlinear pair coder
def compute_hcr_pair_evidence(
    samples_df: pd.DataFrame,
    parents_by_endpoint: Dict[str, List[str]],
    train_patient_ids: set,
) -> dict:
 
    if "patient_id" not in samples_df.columns:
        raise ValueError("samples_df musi zawierac kolumne 'patient_id'.")

    train_mask = samples_df["patient_id"].isin(train_patient_ids)
    train_df = samples_df.loc[train_mask]
    n_train = len(train_df)
    if n_train < 2:
        raise ValueError("Za malo pacjentow treningowych do policzenia HCR (potrzeba >= 2).")
    N = n_train
    n_patients = len(samples_df)

    endpoint_order = list(parents_by_endpoint.keys())
    max_parents = max((len(p) for p in parents_by_endpoint.values()), default=0)
    if max_parents == 0:
        raise ValueError("Zaden endpoint nie ma zadnych binarnych rodzicow - sprawdz parents_by_endpoint.")

    features = np.zeros((n_patients, len(endpoint_order), max_parents, N_PAIR_EVIDENCE_CHANNELS), dtype=np.float32)
    mask = np.zeros((n_patients, len(endpoint_order), max_parents), dtype=np.float32)

    train_positions = np.flatnonzero(train_mask.to_numpy())

    for e_idx, endpoint in enumerate(endpoint_order):
        parents = parents_by_endpoint[endpoint]
        if not parents:
            continue
        if endpoint not in samples_df.columns:
            raise ValueError(f"Brak kolumny endpointu '{endpoint}' w samples_df.")

        y_train = train_df[endpoint].to_numpy(dtype=float)
        p_e_full = float(y_train.mean())

        for r_idx, r in enumerate(parents):
            if r not in samples_df.columns:
                raise ValueError(f"Brak kolumny rodzica '{r}' w samples_df (endpoint={endpoint}).")

            x_train = train_df[r].to_numpy(dtype=float)
            p_r_full = float(x_train.mean())
            n11 = float((x_train * y_train).sum())

            # --- LOO (train): ta sama, zweryfikowana tabela kontyngencji
            # dla WSZYSTKICH 5 statystyk naraz (a11/nmi/OR/RD/lift spojne,
            # bo pochodza z tej samej p_r_loo/p_e_loo/p11_loo per pacjent) ---
            p_r_loo = (N * p_r_full - x_train) / (N - 1)
            p_e_loo = (N * p_e_full - y_train) / (N - 1)
            p11_loo = (n11 - x_train * y_train) / (N - 1)
            stats_loo = _classical4_from_contingency(p_r_loo, p_e_loo, p11_loo)

            denom_loo_x = np.sqrt(np.clip(p_r_loo * (1 - p_r_loo), 1e-12, None))
            phi1_train = (x_train - p_r_loo) / denom_loo_x

            # --- pelne (nie-LOO), dla valid/test - stale, jedna wartosc ---
            p11_full = float((x_train * y_train).mean())
            stats_full_dict = _classical4_from_contingency(
                np.array([p_r_full]), np.array([p_e_full]), np.array([p11_full])
            )
            stats_full = {k: float(v[0]) for k, v in stats_full_dict.items()}
            support_val = float(N / n_patients) if n_patients else 0.0

            # Zloz kanaly per pacjent: najpierw wypelnij WSZYSTKICH pacjentow
            # wartoscia "pelna" (poprawna dla valid/test), potem NADPISZ
            # pozycje treningowe wersja LOO (analogicznie do compute_hcr_wide_scores).
            x_all = samples_df[r].to_numpy(dtype=float)
            phi1_all = (x_all - p_r_full) / np.sqrt(max(p_r_full * (1 - p_r_full), 1e-12))

            block = np.zeros((n_patients, N_PAIR_EVIDENCE_CHANNELS), dtype=np.float32)
            block[:, 0] = phi1_all
            block[:, 1] = stats_full["a11"]
            block[:, 2] = stats_full["nmi"]
            block[:, 3] = stats_full["scaled_log_or"]
            block[:, 4] = stats_full["rd"]
            block[:, 5] = stats_full["scaled_log_lift"]
            block[:, 6] = p_r_full
            block[:, 7] = p_e_full
            block[:, 8] = support_val

            block[train_positions, 0] = phi1_train
            block[train_positions, 1] = stats_loo["a11"]
            block[train_positions, 2] = stats_loo["nmi"]
            block[train_positions, 3] = stats_loo["scaled_log_or"]
            block[train_positions, 4] = stats_loo["rd"]
            block[train_positions, 5] = stats_loo["scaled_log_lift"]
            block[train_positions, 6] = p_r_loo
            block[train_positions, 7] = p_e_loo
            # support (8) i marginesy pelne (6,7) dla valid/test zostaja jak wyzej
            block[:, 9] = block[:, 1] * block[:, 0] # a11 * phi1_patient

            features[:, e_idx, r_idx, :] = block
            mask[:, e_idx, r_idx] = 1.0

    return {
        "endpoint_order": endpoint_order,
        "features": features,
        "mask": mask,
        "patient_id": samples_df["patient_id"].to_numpy(),
    }


# --- "enriched 40D" version (+ 1 activation channel for patient) ---
PAIR_EVIDENCE_CHANNELS_40D = (
    [f"matrix_{i}" for i in range(16)]
    + ["energy_total", "energy_mean", "energy_low_order_frac", "energy_max_abs"]
    + ["marg_U_prevalence", "marg_U_entropy", "marg_U_reserved1", "marg_U_reserved2"]
    + ["marg_V_prevalence", "marg_V_entropy", "marg_V_reserved1", "marg_V_reserved2"]
    + ["joint_activity", "excess_over_independence", "log_lift", "rarity"]
    + ["support", "estimability_mask"]
    + ["type_U_binary", "type_U_count", "type_U_continuous"]
    + ["type_V_binary", "type_V_count", "type_V_continuous"]
    + ["phi1_patient"]
    + ["s_evidence"]
)
N_PAIR_EVIDENCE_CHANNELS_40D = len(PAIR_EVIDENCE_CHANNELS_40D)
assert N_PAIR_EVIDENCE_CHANNELS_40D == 42, N_PAIR_EVIDENCE_CHANNELS_40D


def _binary_entropy(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def compute_hcr_pair_evidence_40d(
    samples_df: pd.DataFrame,
    parents_by_endpoint: Dict[str, List[str]],
    train_patient_ids: set,
    edges: Optional[pd.DataFrame] = None,
) -> dict:
  
    reliability = get_edge_reliability_weights(edges, parents_by_endpoint) if edges is not None else None
    if "patient_id" not in samples_df.columns:
        raise ValueError("samples_df musi zawierac kolumne 'patient_id'.")

    train_mask = samples_df["patient_id"].isin(train_patient_ids)
    train_df = samples_df.loc[train_mask]
    n_train = len(train_df)
    if n_train < 2:
        raise ValueError("Za malo pacjentow treningowych do policzenia HCR (potrzeba >= 2).")
    N = n_train
    n_patients = len(samples_df)

    endpoint_order = list(parents_by_endpoint.keys())
    max_parents = max((len(p) for p in parents_by_endpoint.values()), default=0)
    if max_parents == 0:
        raise ValueError("Zaden endpoint nie ma zadnych binarnych rodzicow - sprawdz parents_by_endpoint.")

    features = np.zeros(
        (n_patients, len(endpoint_order), max_parents, N_PAIR_EVIDENCE_CHANNELS_40D), dtype=np.float32
    )
    mask = np.zeros((n_patients, len(endpoint_order), max_parents), dtype=np.float32)
    train_positions = np.flatnonzero(train_mask.to_numpy())

    for e_idx, endpoint in enumerate(endpoint_order):
        parents = parents_by_endpoint[endpoint]
        if not parents:
            continue
        if endpoint not in samples_df.columns:
            raise ValueError(f"Brak kolumny endpointu '{endpoint}' w samples_df.")

        y_train = train_df[endpoint].to_numpy(dtype=float)
        p_e_full = float(y_train.mean())

        for r_idx, r in enumerate(parents):
            if r not in samples_df.columns:
                raise ValueError(f"Brak kolumny rodzica '{r}' w samples_df (endpoint={endpoint}).")

            x_train = train_df[r].to_numpy(dtype=float)
            p_r_full = float(x_train.mean())
            n11 = float((x_train * y_train).sum())
            p11_full = float((x_train * y_train).mean())

            # --- LOO per pacjent treningowy (ta sama, zweryfikowana tabela
            # kontyngencji co w compute_hcr_pair_evidence) ---
            p_r_loo = (N * p_r_full - x_train) / (N - 1)
            p_e_loo = (N * p_e_full - y_train) / (N - 1)
            p11_loo = (n11 - x_train * y_train) / (N - 1)

            def build_block(p_r, p_e, p11, phi1, n_rows: int) -> np.ndarray:
                """p_r,p_e,p11,phi1: skalary (walidacja/test) lub tablice
                dlugosci n_rows (LOO, train). Zwraca [n_rows, 41]."""
                p_r_arr = np.broadcast_to(p_r, (n_rows,)).astype(np.float64)
                p_e_arr = np.broadcast_to(p_e, (n_rows,)).astype(np.float64)
                p11_arr = np.broadcast_to(p11, (n_rows,)).astype(np.float64)
                stats = _classical4_from_contingency(p_r_arr, p_e_arr, p11_arr)

                block = np.zeros((n_rows, N_PAIR_EVIDENCE_CHANNELS_40D), dtype=np.float32)
                # 0-15: macierz, aktywne tylko [0,0]=a11
                block[:, 0] = stats["a11"]
                # 16-19: energia (dla binary-binary: 1 aktywny wspolczynnik)
                energy_total = stats["a11"] ** 2
                block[:, 16] = energy_total
                block[:, 17] = energy_total  # mean = total, bo tylko 1 aktywny
                block[:, 18] = 1.0            # caly "budzet" energii w rzedzie niskim
                block[:, 19] = np.abs(stats["a11"])
                # 20-23: marginesy U (rodzic)
                block[:, 20] = p_r_arr
                block[:, 21] = _binary_entropy(p_r_arr)
                # 22,23 rezerwa = 0 (miejsce dla przyszlych continuous/count)
                # 24-27: marginesy V (endpoint)
                block[:, 24] = p_e_arr
                block[:, 25] = _binary_entropy(p_e_arr)
                # 28-31: joint activity / rzadkosc
                block[:, 28] = p11_arr
                block[:, 29] = p11_arr - p_r_arr * p_e_arr  # nadwyzka nad niezaleznoscia (licznik a11)
                block[:, 30] = stats["scaled_log_lift"]
                block[:, 31] = np.minimum(p_r_arr, p_e_arr)  # rzadkosc rzadszej strony pary
                # 32: support (stale - nie zalezy od pacjenta)
                block[:, 32] = float(N / n_patients) if n_patients else 0.0
                # 33: maska estymowalnosci (stala dla calego bloku - wlasnosc
                # PARY, nie pojedynczego pacjenta) - liczona na PELNYM train,
                # bo pyta "czy w ogole mamy dosc danych", nie o etykiete pacjenta
                estimable = 1.0 if (1e-3 < p_r_full < 1 - 1e-3 and 1e-3 < p_e_full < 1 - 1e-3) else 0.0
                block[:, 33] = estimable
                # 34-39: one-hot typow (dzis zawsze binary-binary)
                block[:, 34] = 1.0  # type_U_binary
                block[:, 37] = 1.0  # type_V_binary
                # 40: aktywacja pacjenta
                block[:, 40] = phi1
                block[:, 41] = block[:, 0] * block[:, 40]   # matrix_0 * phi1_patient
                return block

            denom_loo_x = np.sqrt(np.clip(p_r_loo * (1 - p_r_loo), 1e-12, None))
            phi1_train = (x_train - p_r_loo) / denom_loo_x
            block_train = build_block(p_r_loo, p_e_loo, p11_loo, phi1_train, n_rows=N)

            x_all = samples_df[r].to_numpy(dtype=float)
            phi1_all = (x_all - p_r_full) / np.sqrt(max(p_r_full * (1 - p_r_full), 1e-12))
            block_full = build_block(p_r_full, p_e_full, p11_full, phi1_all, n_rows=n_patients)

            features[:, e_idx, r_idx, :] = block_full
            features[train_positions, e_idx, r_idx, :] = block_train
            mask[:, e_idx, r_idx] = reliability[endpoint][r] if reliability is not None else 1.0

    return {
        "endpoint_order": endpoint_order,
        "features": features,
        "mask": mask,
        "patient_id": samples_df["patient_id"].to_numpy(),
    }


def compute_hcr_pair_evidence_mixed(
    samples_df: pd.DataFrame,
    parents_by_endpoint: Dict[str, List[Tuple[str, str]]],
    train_patient_ids: set,
    edges: Optional[pd.DataFrame] = None,
    seed: int = 0,
) -> dict:

    if "patient_id" not in samples_df.columns:
        raise ValueError("samples_df musi zawierac kolumne 'patient_id'.")

    train_mask = samples_df["patient_id"].isin(train_patient_ids)
    train_df = samples_df.loc[train_mask]
    n_train = len(train_df)
    if n_train < 2:
        raise ValueError("Za malo pacjentow treningowych do policzenia HCR (potrzeba >= 2).")
    n_patients = len(samples_df)
    train_positions = np.flatnonzero(train_mask.to_numpy())
    all_patient_ids = samples_df["patient_id"].to_numpy()
    train_patient_ids_arr = train_df["patient_id"].to_numpy()

    reliability = get_edge_reliability_weights(edges, parents_by_endpoint) if edges is not None else None

    endpoint_order = list(parents_by_endpoint.keys())
    max_parents = max((len(p) for p in parents_by_endpoint.values()), default=0)
    if max_parents == 0:
        raise ValueError("Zaden endpoint nie ma zadnych rodzicow - sprawdz parents_by_endpoint.")

    type_onehot = {"binary": (1.0, 0.0, 0.0), "count": (0.0, 1.0, 0.0), "continuous": (0.0, 0.0, 1.0)}

    features = np.zeros(
        (n_patients, len(endpoint_order), max_parents, N_PAIR_EVIDENCE_CHANNELS_40D), dtype=np.float32
    )
    mask = np.zeros((n_patients, len(endpoint_order), max_parents), dtype=np.float32)

    for e_idx, endpoint in enumerate(endpoint_order):
        parents = parents_by_endpoint[endpoint]
        if not parents:
            continue
        if endpoint not in samples_df.columns:
            raise ValueError(f"Brak kolumny endpointu '{endpoint}' w samples_df.")

        y_train = train_df[endpoint].to_numpy(dtype=float)
        p_e_full = float(y_train.mean())

        for r_idx, (r, r_type) in enumerate(parents):
            if r not in samples_df.columns:
                raise ValueError(f"Brak kolumny rodzica '{r}' w samples_df (endpoint={endpoint}).")

            x_train = train_df[r].to_numpy(dtype=float)
            x_all = samples_df[r].to_numpy(dtype=float)

            if r_type == "binary":
                p_r_full = float(x_train.mean())
                basis_train = ((x_train - p_r_full) / np.sqrt(max(p_r_full * (1 - p_r_full), 1e-12)))[:, None]
                basis_all = ((x_all - p_r_full) / np.sqrt(max(p_r_full * (1 - p_r_full), 1e-12)))[:, None]
                marg1, marg2 = p_r_full, _binary_entropy(np.array([p_r_full]))[0]
            elif r_type in ("count", "continuous"):
                if r_type == "continuous":
                    u_train = _train_ecdf_transform(x_train, x_train)
                    u_all = _train_ecdf_transform(x_train, x_all)
                else:  # count
                    u_train = _count_to_uniform(x_train, x_train, train_patient_ids_arr, r, seed)
                    u_all = _count_to_uniform(x_train, x_all, all_patient_ids, r, seed)
                basis_train = _legendre_psi(u_train, degree=4)
                basis_all = _legendre_psi(u_all, degree=4)
                # Skosnosc (train-only) jako bezskalowy deskryptor ksztaltu
                # marginesu - pozycja 2 (kurtoza) zarezerwowana na przyszlosc.
                x_std = x_train.std()
                skew = float(((x_train - x_train.mean()) ** 3).mean() / (x_std ** 3 + 1e-12)) if x_std > 1e-8 else 0.0
                marg1, marg2 = skew, 0.0
            else:
                raise ValueError(f"Nieznany value_type={r_type!r} dla rodzica '{r}'.")

            a_loo, a_full = _pairwise_coefficients_loo(y_train, basis_train, train_positions, n_patients)
            K = basis_train.shape[1]

            energy_full = float((a_full ** 2).sum())
            energy_loo = (a_loo ** 2).sum(axis=1)  # [N_train]

            block = np.zeros((n_patients, N_PAIR_EVIDENCE_CHANNELS_40D), dtype=np.float32)

            block[:, 0:K] = a_full[None, :]
            block[:, 16] = energy_full
            block[:, 17] = energy_full / max(K, 1)
            block[:, 18] = (a_full[0] ** 2) / energy_full if energy_full > 1e-12 else 0.0
            block[:, 19] = float(np.abs(a_full).max()) if K else 0.0
            block[:, 20] = marg1
            block[:, 21] = marg2
            block[:, 24] = p_e_full
            block[:, 25] = _binary_entropy(np.array([p_e_full]))[0]
            block[:, 32] = float(n_train / n_patients) if n_patients else 0.0
            block[:, 33] = 1.0
            block[:, 34:37] = type_onehot[r_type]
            block[:, 37:40] = type_onehot["binary"]  # V (endpoint) zawsze binarny
            block[:, 40] = basis_all[:, 0]  # aktywacja pacjenta = 1. wspolrzedna bazy
  
            block[train_positions, 0:K] = a_loo
            block[train_positions, 16] = energy_loo
            block[train_positions, 17] = energy_loo / max(K, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                block[train_positions, 18] = np.where(energy_loo > 1e-12, (a_loo[:, 0] ** 2) / energy_loo, 0.0)
            block[train_positions, 19] = np.abs(a_loo).max(axis=1) if K else 0.0
            block[train_positions, 40] = basis_train[:, 0]
            block[:, 41] = block[:, 0] * block[:, 40]
            

            features[:, e_idx, r_idx, :] = block
            mask_val = (reliability[endpoint][r] if reliability is not None else 1.0)
            mask[:, e_idx, r_idx] = mask_val

    return {
        "endpoint_order": endpoint_order,
        "features": features,
        "mask": mask,
        "patient_id": samples_df["patient_id"].to_numpy(),
    }


def compute_hcr_pair_evidence_by_node_type(
    samples_df: pd.DataFrame,
    parents_by_endpoint_by_type: Dict[str, Dict[str, List[Tuple[str, str]]]],
    train_patient_ids: set,
    edges: Optional[pd.DataFrame] = None,
    seed: int = 0,
) -> Dict[str, Optional[dict]]:

    out: Dict[str, Optional[dict]] = {}
    for nt in PARENT_NODE_TYPES:
        parents_this_type = {
            ep: by_type.get(nt, [])
            for ep, by_type in parents_by_endpoint_by_type.items()
        }
        has_any = any(len(v) > 0 for v in parents_this_type.values())
        if not has_any:
            out[nt] = None
            continue
        out[nt] = compute_hcr_pair_evidence_mixed(
            samples_df, parents_this_type, train_patient_ids, edges=edges, seed=seed
        )
    return out


def compute_hcr_wide_scores(
    samples_df: pd.DataFrame,
    parents_by_endpoint: Dict[str, List[str]],
    train_patient_ids: set,
) -> pd.DataFrame:

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

            denom_loo_x = np.sqrt(np.clip(p_r_loo * (1 - p_r_loo), 1e-12, None))
            phi_x_train_loo = (x_train - p_r_loo) / denom_loo_x
            score_train_loo += a_loo * phi_x_train_loo

        # scal: score_full juz ma poprawne wartosci dla valid/test (pelny a_full);
        # dla train nadpisujemy wersja LOO
        train_idx_positions = np.flatnonzero(train_mask.to_numpy())
        score_full[train_idx_positions] = score_train_loo

        out[endpoint] = score_full

    return out

#technical helper: keep only parents that have a column in samples_df (added for proxy model, where some parents are not observed in the data)
def keep_observed(parents_by_endpoint, samples_df):

    observed = set(samples_df.columns)

    def _name(p):
        return p[0] if isinstance(p, (tuple, list)) else p

    out = {}
    for ep, parents in parents_by_endpoint.items():
        if isinstance(parents, dict):
            out[ep] = {t: [p for p in lst if _name(p) in observed] for t, lst in parents.items()}
        else:
            out[ep] = [p for p in parents if _name(p) in observed]
    return out