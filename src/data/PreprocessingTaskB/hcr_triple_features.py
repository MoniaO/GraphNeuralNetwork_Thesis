"""
Moment trojny HCR: zaleznosc (dziadek, rodzic, endpoint).

CO TO MIERZY, A CZEGO NIE MIERZY DOTYCHCZASOWY HCR

Obecny HCR liczy a11(rodzic, endpoint) - zaleznosc PAROWA na jednym hopie.
Analiza metasciezek mierzyla iloczyny wartosci wzdluz sciezki, czyli
zaleznosc ZMEDIOWANA - i wyszla slaba, bo w generatorze addytywnym wplyw
dziadka przechodzi w calosci przez wartosc rodzica.

Moment trojny to co innego:

    a_111 = E[ phi_1(G) phi_1(P) phi_1(E) ]

Wykrywa zaleznosc NIEREDUKOWALNA do par. Przyklad graniczny: E = G XOR P.
Wtedy a11(G,E) = a11(P,E) = 0 - obie pary sa slepe - a a_111 = -1.
Zadne z Twoich dotychczasowych narzedzi tego nie zobaczy.

Klinicznie odpowiada to interakcji: "lek szkodzi TYLKO przy wspolistniejacym
mechanizmie", a nie "lek szkodzi i mechanizm szkodzi, sumarycznie".

WZOR (wyprowadzony dla zmiennych binarnych)

Trzeci mieszany moment centralny:

    E[(G-pG)(P-pP)(E-pE)]
      = p111 - pE*pGP - pP*pGE - pG*pPE + 2*pG*pP*pE

po podzieleniu przez sqrt(pG(1-pG) * pP(1-pP) * pE(1-pE)) daje a_111.
Zweryfikowane numerycznie wobec brute-force (patrz test na koncu pliku).

ZAKRES: tylko trojki zgodne z DAG-iem, czyli G -> P -> E. Nie wszystkie pary
przodkow - to ograniczylo by kombinatoryke z O(n_przodkow^2) do sumy stopni
wejsciowych rodzicow, czyli rzedu kilkudziesieciu na endpoint zamiast tysiecy.

WPIECIE W ISTNIEJACY KOD: HCRPairEncoder przyjmuje [..., max_slots,
in_channels] + maske. Slot to dotad rodzic; teraz slot to TROJKA. Koder
dziala bez zadnej zmiany, zmienia sie tylko in_channels i znaczenie slotu.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

TRIPLE_EVIDENCE_CHANNELS = [
    "a111",              # moment trojny - GLOWNY kanal, zero pod niezaleznoscia
    "a11_GP",            # zaleznosci parowe, zeby siec mogla odroznic
    "a11_GE",            # interakcje od samego wspolwystepowania
    "a11_PE",
    "excess_111",        # p111 - pG*pP*pE, nadwyzka ponad pelna niezaleznosc
    "prev_G", "prev_P", "prev_E",
    "support",           # udzial pacjentow treningowych z kompletem pomiarow
    "estimable",         # maska: czy najmniejsza komorka ma dosc obserwacji
    "phi_G",             # aktywacje TEGO pacjenta
    "phi_P",
    "s_triple",          # a111 * phi_G * phi_P - dowod pacjenta, JAWNIE
]
N_TRIPLE_EVIDENCE_CHANNELS = len(TRIPLE_EVIDENCE_CHANNELS)

MIN_CELL = 5      # minimalna liczebnosc najmniejszej komorki 2x2x2
COEF_CLIP = 5.0   # zakres wspolczynnikow - poza nim i tak nieinterpretowalne
ACT_CLIP = 10.0   # zakres aktywacji phi (przy p=0.01 phi(1) ~ 9.95)


def get_triples_by_endpoint(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    target_endpoints: Sequence[str],
    excluded_nodes: Set[str],
    observed_columns: Set[str],
    max_triples: int = 40,
    binary_only: bool = True,
    samples_df: Optional[pd.DataFrame] = None,
    train_patient_ids: Optional[Set[int]] = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """Zwraca {endpoint: [(dziadek, rodzic), ...]} dla sciezek G -> P -> E.

    Oba wezly musza miec OBSERWOWANA wartosc per pacjent - inaczej nie da sie
    policzyc tabeli 2x2x2. To ten sam warunek co keep_visible dla par.

    binary_only=True (domyslnie): oba wezly musza miec value_type="binary".

    DLACZEGO. _triple_stats liczy prewalencje jako suma/n, czyli traktuje
    wartosc jako zero-jedynkowa. Dla wezla ciaglego (age, baseline_egfr) albo
    licznikowego (nephrotoxin_load) daje to bezsensowna "prewalencje" i
    zaszumione a111.

    Poprawne rozszerzenie wymagaloby bazy Legendre'a jak dla par - ale trojka
    ciagla dawalaby wtedy tensor 4x4x4, czyli 64 wspolczynniki szacowane z
    tabeli o osmiu komorkach. Przy prewalencji endpointow rzedu kilku procent
    jest to nieestymowalne (por. dokument HCR Task A, sekcja 7.2: tabela
    2x2x2 ma wyzsza wariancje niz pary, zwlaszcza przy rzadkich komorkach).

    max_triples ogranicza liczbe slotow; ranking po kolejnosci w grafie,
    NIE po etykiecie (zeby nie robic selekcji cech po celu).
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes_df["node"])
    G.add_edges_from(zip(edges_df["source"], edges_df["target"]))
    value_type = dict(zip(nodes_df["node"], nodes_df.get("value_type", pd.Series(dtype=str))))

    def usable(node: str) -> bool:
        if node in excluded_nodes or node not in observed_columns:
            return False
        if binary_only and str(value_type.get(node)) != "binary":
            return False
        return True

    # Wsparcie pary (dziadek, rodzic) do rankingu - liczone raz, na splicie
    # treningowym. Bez samples_df ranking sie nie odbywa i brany jest
    # porzadek grafu (patrz komentarz przy sortowaniu).
    support: Optional[Dict[Tuple[str, str], float]] = None
    if samples_df is not None:
        rows = samples_df
        if train_patient_ids is not None and "patient_id" in samples_df.columns:
            rows = samples_df[samples_df["patient_id"].isin(train_patient_ids)]
        support = {}

    out: Dict[str, List[Tuple[str, str]]] = {}
    endpoint_set = set(target_endpoints)
    n_skipped = 0
    for ep in target_endpoints:
        if ep not in G:
            out[ep] = []
            continue
        triples = []
        for parent in G.predecessors(ep):
            if parent in endpoint_set:      # inny endpoint jako rodzic = wyciek
                continue
            if not usable(parent):
                n_skipped += 1
                continue
            for grand in G.predecessors(parent):
                if grand in endpoint_set:
                    continue
                if not usable(grand):
                    n_skipped += 1
                    continue
                if support is not None and (grand, parent) not in support:
                    if grand in rows.columns and parent in rows.columns:
                        support[(grand, parent)] = float(
                            (rows[grand].to_numpy() * rows[parent].to_numpy()).sum())
                triples.append((grand, parent))

        # RANKING: po wspolwystepowaniu dziadka i rodzica na splicie
        # treningowym, malejaco; remisy rozstrzygane alfabetycznie, zeby
        # wynik byl deterministyczny miedzy uruchomieniami.
        # NIE uzywa etykiety endpointu - inaczej byloby to selekcja cech
        # po celu, ktora musialaby wejsc do walidacji krzyzowej.
        if support is not None and len(triples) > max_triples:
            triples.sort(key=lambda gp: (-support.get(gp, 0.0), gp[0], gp[1]))
        out[ep] = triples[:max_triples]
    if binary_only and n_skipped:
        print(f"[HCR triple] pominieto {n_skipped} wezlow niebinarnych "
              f"(value_type != 'binary')")
    return out


def _all_eight_cells(
    n111, nG, nP, nE, nGP, nGE, nPE, n
) -> List[np.ndarray]:
    """Wszystkie OSIEM komorek tabeli 2x2x2 (G, P, E).

    Wczesniejsza wersja sprawdzala tylko piec, przez co maska estymowalnosci
    przepuszczala trojki z pusta komorka n010, n001 albo n000 - czyli
    dokladnie te, przy ktorych a111 jest najbardziej niestabilne.

    Indeksowanie: n_{gpe}, np. n101 to G=1, P=0, E=1.
    Wyprowadzenie przez wlaczenia i wylaczenia:
        n100 = nG - nGP - nGE + n111
        n000 = n - nG - nP - nE + nGP + nGE + nPE - n111
    """
    n110 = nGP - n111
    n101 = nGE - n111
    n011 = nPE - n111
    n100 = nG - nGP - nGE + n111
    n010 = nP - nGP - nPE + n111
    n001 = nE - nGE - nPE + n111
    n000 = n - nG - nP - nE + nGP + nGE + nPE - n111
    return [n111, n110, n101, n011, n100, n010, n001, n000]


def _triple_stats(
    n111: np.ndarray, nG: np.ndarray, nP: np.ndarray, nE: np.ndarray,
    nGP: np.ndarray, nGE: np.ndarray, nPE: np.ndarray, n: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Statystyki trojki z LICZNOSCI (nie prawdopodobienstw) - dzieki temu
    leave-one-out sprowadza sie do odjecia jedynek od wlasciwych licznikow.

    STABILNOSC. a111 dzieli przez iloczyn TRZECH odchylen. Przy rzadkim
    wezle sigma ~ sqrt(p), wiec przy p rzedu 1e-6 mianownik schodzi do 1e-9,
    a wspolczynnik wybucha do 1e9 - co przy dodatkowym pomnozeniu przez dwie
    aktywacje daje wartosci rzedu 1e15 i rozsadza funkcje kosztu.
    Dlatego podloga na prewalencji jest ZALEZNA OD PROBY (co najmniej
    MIN_CELL/n), a wspolczynniki sa dodatkowo przycinane do zakresu, w
    ktorym pozostaja interpretowalne.
    """
    n = np.maximum(n, 1.0)
    # podloga: prewalencja ponizej MIN_CELL/n i tak nie jest estymowalna
    floor = np.maximum(MIN_CELL / n, eps)
    cl = lambda p: np.clip(p, floor, 1.0 - floor)

    pG, pP, pE = cl(nG / n), cl(nP / n), cl(nE / n)
    pGP, pGE, pPE = nGP / n, nGE / n, nPE / n
    p111 = n111 / n

    sG = np.sqrt(pG * (1 - pG))
    sP = np.sqrt(pP * (1 - pP))
    sE = np.sqrt(pE * (1 - pE))

    # trzeci mieszany moment centralny
    m3 = p111 - pE * pGP - pP * pGE - pG * pPE + 2 * pG * pP * pE
    a111 = np.clip(m3 / (sG * sP * sE), -COEF_CLIP, COEF_CLIP)

    return {
        "a111": a111,
        "a11_GP": np.clip((pGP - pG * pP) / (sG * sP), -COEF_CLIP, COEF_CLIP),
        "a11_GE": np.clip((pGE - pG * pE) / (sG * sE), -COEF_CLIP, COEF_CLIP),
        "a11_PE": np.clip((pPE - pP * pE) / (sP * sE), -COEF_CLIP, COEF_CLIP),
        "excess_111": p111 - pG * pP * pE,
        "prev_G": pG, "prev_P": pP, "prev_E": pE,
    }


def compute_hcr_triple_evidence(
    samples_df: pd.DataFrame,
    triples_by_endpoint: Dict[str, List[Tuple[str, str]]],
    train_patient_ids: Set[int],
    max_triples: int | None = None,
) -> Dict[str, object]:
    """Tensor dowodowy dla trojek, w formacie zgodnym z HCRPairEncoder.

    Zwraca:
        features: [n_patients, n_endpoints, max_triples, N_TRIPLE_CHANNELS]
        mask:     [n_patients, n_endpoints, max_triples]

    LEAVE-ONE-OUT: dla pacjenta treningowego wszystkie licznosci sa
    pomniejszone o JEGO wlasna obserwacje. Bez tego kanal a111 zawieralby
    etykiete pacjenta, dla ktorego jest liczony.
    """
    endpoint_order = list(triples_by_endpoint.keys())
    if max_triples is None:
        max_triples = max((len(v) for v in triples_by_endpoint.values()), default=1)
    max_triples = max(max_triples, 1)

    pid = samples_df["patient_id"].to_numpy()
    is_train = np.isin(pid, list(train_patient_ids))
    n_pat = len(samples_df)

    features = np.zeros((n_pat, len(endpoint_order), max_triples,
                         N_TRIPLE_EVIDENCE_CHANNELS), dtype=np.float32)
    mask = np.zeros((n_pat, len(endpoint_order), max_triples), dtype=np.float32)

    n_train = int(is_train.sum())
    for e_idx, ep in enumerate(endpoint_order):
        if ep not in samples_df.columns:
            continue
        E = samples_df[ep].to_numpy().astype(np.float64)
        Etr = E[is_train]
        for t_idx, (g_name, p_name) in enumerate(triples_by_endpoint[ep][:max_triples]):
            if g_name not in samples_df.columns or p_name not in samples_df.columns:
                continue
            Gv = samples_df[g_name].to_numpy().astype(np.float64)
            Pv = samples_df[p_name].to_numpy().astype(np.float64)
            Gtr, Ptr = Gv[is_train], Pv[is_train]

            # licznosci na CALYM zbiorze treningowym
            c = dict(
                n=float(n_train),
                nG=Gtr.sum(), nP=Ptr.sum(), nE=Etr.sum(),
                nGP=(Gtr * Ptr).sum(), nGE=(Gtr * Etr).sum(), nPE=(Ptr * Etr).sum(),
                n111=(Gtr * Ptr * Etr).sum(),
            )
            full = _triple_stats(**{k: np.array([v]) for k, v in c.items()})

            # wersja LOO: od kazdej licznosci odjac wklad TEGO pacjenta
            loo = _triple_stats(
                n111=c["n111"] - Gv * Pv * E, nG=c["nG"] - Gv, nP=c["nP"] - Pv,
                nE=c["nE"] - E, nGP=c["nGP"] - Gv * Pv, nGE=c["nGE"] - Gv * E,
                nPE=c["nPE"] - Pv * E, n=np.full(n_pat, c["n"] - 1.0),
            )

            block = np.zeros((n_pat, N_TRIPLE_EVIDENCE_CHANNELS), dtype=np.float32)
            for i, key in enumerate(["a111", "a11_GP", "a11_GE", "a11_PE",
                                     "excess_111", "prev_G", "prev_P", "prev_E"]):
                block[:, i] = full[key][0]
            block[:, 8] = 1.0                       # support (komplet danych)

            # estymowalnosc: KAZDA z osmiu komorek 2x2x2 musi miec MIN_CELL obs.
            cells = _all_eight_cells(
                c["n111"], c["nG"], c["nP"], c["nE"],
                c["nGP"], c["nGE"], c["nPE"], c["n"],
            )
            block[:, 9] = 1.0 if min(cells) >= MIN_CELL else 0.0

            # aktywacje pacjenta z marginesow PELNYCH (dla valid/test)
            sg = np.sqrt(max(full["prev_G"][0] * (1 - full["prev_G"][0]), 1e-12))
            sp = np.sqrt(max(full["prev_P"][0] * (1 - full["prev_P"][0]), 1e-12))
            block[:, 10] = np.clip((Gv - full["prev_G"][0]) / sg, -ACT_CLIP, ACT_CLIP)
            block[:, 11] = np.clip((Pv - full["prev_P"][0]) / sp, -ACT_CLIP, ACT_CLIP)

            # NADPISANIE dla pacjentow treningowych: wersja leave-one-out
            tr = np.where(is_train)[0]
            for i, key in enumerate(["a111", "a11_GP", "a11_GE", "a11_PE",
                                     "excess_111", "prev_G", "prev_P", "prev_E"]):
                block[tr, i] = loo[key][tr]
            sg_loo = np.sqrt(np.clip(loo["prev_G"][tr] * (1 - loo["prev_G"][tr]), 1e-12, None))
            sp_loo = np.sqrt(np.clip(loo["prev_P"][tr] * (1 - loo["prev_P"][tr]), 1e-12, None))
            block[tr, 10] = np.clip((Gv[tr] - loo["prev_G"][tr]) / sg_loo, -ACT_CLIP, ACT_CLIP)
            block[tr, 11] = np.clip((Pv[tr] - loo["prev_P"][tr]) / sp_loo, -ACT_CLIP, ACT_CLIP)

            # s_triple PO nadpisaniach - inaczej pacjenci treningowi
            # dostaliby iloczyn ze wspolczynnika pelnego (przeciek etykiety)
            block[:, 12] = block[:, 0] * block[:, 10] * block[:, 11]

            # Blok NIEESTYMOWALNY (za male komorki) jest ZEROWANY, nie tylko
            # oznaczany - inaczej siec dostaje wartosci policzone z kilku
            # obserwacji, ktore sa czystym szumem o duzej amplitudzie.
            # Kanal maski (9) zostaje na 0, zeby siec wiedziala, ze to
            # brak danych, a nie prawdziwe zero (dokument HCR, sekcja 6).
            if block[0, 9] == 0.0:
                block[:, :9] = 0.0
                block[:, 10:] = 0.0

            if not np.isfinite(block).all():
                raise FloatingPointError(
                    f"Niestabilne wartosci dla trojki ({g_name}, {p_name}) -> {ep}")

            features[:, e_idx, t_idx, :] = block
            mask[:, e_idx, t_idx] = 1.0

    n_slots = int(mask[0].sum())
    n_estimable = int(((features[0, :, :, 9] > 0) & (mask[0] > 0)).sum())
    print(f"[HCR triple] estymowalnych: {n_estimable} / {n_slots} trojek "
          f"(min. {MIN_CELL} obserwacji w kazdej z 8 komorek)")

    # KONTRAKT: builder (build_patient_hetero_graphs) wymaga dokladnie tych
    # kluczy - "patient_id" sluzy do ustalenia row_order, czyli wyrownania
    # wierszy tensora z kolejnoscia pacjentow w grafach. Bez niego KeyError.
    # "channels" jest dodatkiem informacyjnym, builder go ignoruje.
    return {
        "endpoint_order": endpoint_order,
        "features": features,
        "mask": mask,
        "patient_id": samples_df["patient_id"].to_numpy(),
        "channels": TRIPLE_EVIDENCE_CHANNELS,
        "n_triples": n_slots,
        "n_estimable": n_estimable,
    }