# %%
"""
==============================================================================
SUFIT BAYESA NA TWOICH PRAWDZIWYCH DANYCH - wersja pod Jupyter (krok po kroku)
==============================================================================
Ten skrypt liczy GORNA GRANICE AUC/AUPRC, jakiej moze osiagnac JAKIKOLWIEK
model na Twoim zbiorze - GNN, boosting, HCR, cokolwiek. Nie jest to
oszacowanie heurystyczne: jest to dokladne prawdopodobienstwo warunkowe,
z jakim Twoj generator losowal etykiete.

RÓŻNICA WZGLĘDEM WERSJI CLI:
- Brak argparse. Wszystkie sciezki i opcje ustawiasz w komorce "KONFIGURACJA"
  ponizej, jak zwykle zmienne.
- Mozesz wskazac TARGET_ENDPOINT (np. "AKI"), zeby policzyc sufit
  TYLKO dla jednego endpointu, zamiast dla wszystkich naraz.
- Kod jest podzielony na komorki (# %%), zeby moc odpalac krok po kroku
  i podgladac posrednie wyniki (np. world.head(), liste endpointow, eta).
==============================================================================
"""

# %%
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

# %%
# ----------------------------- KONFIGURACJA ---------------------------------
# Podmien na swoje sciezki.
WORLD_PATH = "/Users/monika/data/synthetic_pharmacotherapy_v3_samples_clean.csv"
NODES_PATH = "/Users/monika/data/synthetic_pharmacotherapy_v3_nodes.csv"
EDGES_PATH = "/Users/monika/data/synthetic_pharmacotherapy_v3_edges_audited.csv"
SPLITS_PATH = "/Users/monika/data/splits/patient_splits_v3.csv"  # albo None, jesli nie chcesz filtrowac po splicie
SPLIT_NAME = "validation"          # ktory split raportowac ("val", "test", ...)
YOUR_AUC = 0.625            # Twoje osiagniete srednie AUC (skala 0-1!), albo None

# Jesli chcesz policzyc sufit TYLKO dla jednego endpointu (np. "AKI"),
# wpisz jego nazwe tutaj. Jesli None, skrypt policzy dla WSZYSTKICH endpointow.
TARGET_ENDPOINT = None    # np. "AKI", "DILI", "QT_arrhythmia" ... albo None

COL_NODE = "node"
COL_NODE_TYPE = "node_type"
COL_BASE_PREV = "base_prevalence"
COL_SRC = "source"
COL_DST = "target"
COL_EFFECT = "effect_size"
COL_SIGN = "effect_sign"
DEFAULT_EFFECT = 0.5  # tak jak w generatorze, gdy effect_size jest NaN
DEFAULT_SIGN = 1.0
ENDPOINT_TYPE = "clinical_endpoint"
# ----------------------------------------------------------------------------

# %%
def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_any(path: str) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if path.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)
    raise ValueError(f"Nieobslugiwany format: {path}")


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC przez statystyke Manna-Whitneya - bez zaleznosci od sklearn."""
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # sredni rank przy remisach
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Uwaga: przy DUZEJ liczbie remisow w s wynik moze roznic sie od
    sklearn.average_precision_score w 3. miejscu po przecinku (inna konwencja
    obslugi remisow). AUC powyzej zgadza sie ze sklearn dokladnie."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())

# %%
# --- KROK 1: wczytanie danych ---
world = read_any(WORLD_PATH)
nodes = read_any(NODES_PATH)
edges = read_any(EDGES_PATH)

print(f"world: {world.shape}, nodes: {nodes.shape}, edges: {edges.shape}")
world.head()

# %%
# --- KROK 2: opcjonalny filtr po splicie ---
if SPLITS_PATH:
    sp = read_any(SPLITS_PATH)
    key = "patient_id"
    split_col = next((c for c in sp.columns if c != key), None)
    keep = sp.loc[sp[split_col].astype(str) == SPLIT_NAME, key]
    before = len(world)
    world = world[world[key].isin(set(keep))].reset_index(drop=True)
    print(f"Split '{SPLIT_NAME}': {len(world):,} z {before:,} pacjentow")

# %%
# --- KROK 3: lista dostepnych endpointow (podejrzyj, zanim wybierzesz TARGET_ENDPOINT) ---
meta = nodes.set_index(COL_NODE).to_dict(orient="index")
all_endpoints = [n for n, m in meta.items()
                 if str(m.get(COL_NODE_TYPE, "")) == ENDPOINT_TYPE and n in world.columns]

if not all_endpoints:
    raise ValueError("Nie znaleziono endpointow. Sprawdz COL_NODE_TYPE / ENDPOINT_TYPE.")

print("Dostepne endpointy:", all_endpoints)

# %%
# --- KROK 4: wybor endpointow do policzenia ---
if TARGET_ENDPOINT is not None:
    if TARGET_ENDPOINT not in all_endpoints:
        raise ValueError(
            f"'{TARGET_ENDPOINT}' nie jest dostepnym endpointem. "
            f"Wybierz jeden z: {all_endpoints}"
        )
    endpoints = [TARGET_ENDPOINT]
else:
    endpoints = all_endpoints

print("Bede liczyc sufit dla:", endpoints)

# %%
# --- KROK 5: indeks krawedzi wchodzacych do kazdego wezla ---
inc: dict[str, list] = {}
for _, r in edges.iterrows():
    inc.setdefault(str(r[COL_DST]), []).append(r)

# %%
# --- KROK 6: glowna petla - liczenie sufitu Bayesa per endpoint ---
print("=" * 108)
print("SUFIT BAYESA - maksymalne osiagalne AUC/AUPRC dla kazdego endpointu")
print("=" * 108)
hdr = (f"{'Endpoint':<30}{'n_poz':>8}{'prewalencja':>13}"
       f"{'AUC sufit':>12}{'AUPRC sufit':>13}{'#rodzicow':>11}")
print(hdr)
print("-" * len(hdr))

rows = []
eta_by_endpoint: dict[str, np.ndarray] = {}  # zachowujemy eta, przydatne do dalszej analizy w notebooku

for ep in endpoints:
    base = meta[ep].get(COL_BASE_PREV, np.nan)
    if not np.isfinite(base):
        print(f"{ep:<30}{'brak base_prevalence - pominieto':>60}")
        continue

    eta = np.full(len(world), logit(float(base)), dtype=float)
    n_par = 0
    missing = []
    for e in inc.get(ep, []):
        src = str(e[COL_SRC])
        if src not in world.columns:
            missing.append(src)
            continue
        eff = float(e[COL_EFFECT]) if pd.notna(e.get(COL_EFFECT)) else DEFAULT_EFFECT
        sgn = float(e[COL_SIGN]) if pd.notna(e.get(COL_SIGN)) else DEFAULT_SIGN
        v = world[src].to_numpy(float)
        fin = v[np.isfinite(v)]
        # DOKLADNIE ta sama standaryzacja co w generatorze
        if len(np.unique(fin)) > 2:
            sd = float(np.std(fin))
            if sd > 1e-8:
                v = (v - float(np.mean(fin))) / sd
        eta += sgn * eff * np.nan_to_num(v)
        n_par += 1

    eta_by_endpoint[ep] = eta
    y = world[ep].to_numpy(float)
    p = sigmoid(eta)
    auc = roc_auc(y, p)
    apr = average_precision(y, p)
    rows.append((ep, auc, apr, y.mean()))
    note = f" (brak w danych: {','.join(missing[:2])})" if missing else ""
    print(f"{ep:<30}{int(y.sum()):>8}{y.mean():>13.4f}"
          f"{auc:>12.4f}{apr:>13.5f}{n_par:>11}{note}")

# %%
# --- KROK 7: podsumowanie i interpretacja ---
valid = [r for r in rows if np.isfinite(r[1])]
if not valid:
    raise ValueError("Brak poprawnie policzonych endpointow - sprawdz dane wejsciowe.")

mean_auc = float(np.mean([r[1] for r in valid]))
print("-" * len(hdr))
print(f"{'SREDNIA (wybrane endpointy)':<30}{'':>8}{'':>13}{mean_auc:>12.4f}")

# srednia po "dobrej osemce" - bez trzech skrajnie rzadkich (pomijana automatycznie,
# jesli akurat liczysz tylko jeden TARGET_ENDPOINT spoza tej listy)
noisy = {"Serotonin_syndrome", "Rhabdomyolysis", "Lactic_acidosis"}
good = [r for r in valid if r[0] not in noisy]
mg = float(np.mean([r[1] for r in good])) if good else mean_auc
if good:
    print(f"{'SREDNIA (bez 3 zaszumionych)':<30}{'':>8}{'':>13}{mg:>12.4f}")

print()
print("=" * 108)
print("INTERPRETACJA")
print("=" * 108)
if YOUR_AUC is not None:
    head_ceiling = mg - 0.5
    head_yours = YOUR_AUC - 0.5
    pct = 100.0 * head_yours / head_ceiling if head_ceiling > 1e-9 else float("nan")
    print(f"""
Sufit Bayesa (dla wybranych endpointow) : {mg:.4f}
Twoj wynik                              : {YOUR_AUC:.4f}
Naddatek dostepny (sufit - 0.5)         : {head_ceiling:.4f}
Naddatek wykorzystany                   : {head_yours:.4f}
---------------------------------------------------------------
WYKORZYSTANIE DOSTEPNEGO SYGNALU        : {pct:.1f}%
""")
    if pct > 90:
        print(" => Model jest praktycznie na granicy identyfikowalnosci.\n"
              "    Dalsza praca nad architektura NIE podniesie AUC.\n"
              "    Aby uzyskac wyzsze AUC, trzeba ZMIENIC GENERATOR\n"
              "    (wieksze effect_size lub wyzsze base_prevalence).")
    elif pct > 70:
        print(" => Model wykorzystuje wiekszosc dostepnego sygnalu.\n"
              "    Przestrzen na poprawe architektury jest waska.")
    else:
        print(" => Zostaje realna przestrzen na poprawe modelu.")
else:
    print(f"""
Ustaw YOUR_AUC = {mg:.3f} w konfiguracji na gorze, aby zobaczyc, jaki procent
dostepnego sygnalu wykorzystuje Twoj model.

UWAGA: jesli sufit dla endpointu wynosi ~0.52-0.55, to ten endpoint jest
W PRAKTYCE NIEPRZEWIDYWALNY przy obecnej parametryzacji generatora i
nie powinien wchodzic do sredniej raportowanej jako glowny wynik.
""")

# %%
# --- KROK 8 (opcjonalny): podglad eta / p dla wybranego endpointu ---
# Przydatne do dalszej analizy w notebooku, np. histogramu eta albo
# porownania z predykcjami Twojego modelu GNN.
if TARGET_ENDPOINT is not None and TARGET_ENDPOINT in eta_by_endpoint:
    eta_target = eta_by_endpoint[TARGET_ENDPOINT]
    p_target = sigmoid(eta_target)
    print(f"eta[{TARGET_ENDPOINT}] - min/mean/max:",
          eta_target.min(), eta_target.mean(), eta_target.max())
    print(f"p_hat[{TARGET_ENDPOINT}] - min/mean/max:",
          p_target.min(), p_target.mean(), p_target.max())
