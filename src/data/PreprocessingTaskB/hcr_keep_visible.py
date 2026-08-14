"""
Filtr rodzicow HCR zgodny z rezimem obserwowalnosci.

PROBLEM. keep_observed sprawdza tylko, czy rodzic ma KOLUMNE w samples_df.
Kolumna istnieje niezaleznie od rezimu, wiec w rezimie 'bedside' HCR czytal
geny bind__ (typ drug_exposure), ktore rezim mial zaslonic przed modelem.
Efekt: bedside + HCR osiagalo 0.754, czyli tyle samo co pelny rezim przy L1
(0.756) - model "ubogi" mial wynik modelu pelnego. Ta liczba mierzyla dostep
do danych, nie architekture.

ROZWIAZANIE. Rodzic HCR musi spelniac TRZY warunki:
  1. ma kolumne w samples_df                  (istnieje obserwowana wartosc)
  2. jego typ jest widoczny w danym rezimie   (REGIME_VISIBLE_TYPES)
  3. nie jest w excluded_nodes                (latentny, potomek endpointu
                                               albo ukryty bezposredni rodzic)

Warunek 3 jest osobny od 1, bo excluded_nodes dziala na etapie budowy grafow
- kolumna w samples_df zostaje, wiec sam warunek 1 by go nie wychwycil. Ma to
znaczenie przy eksperymencie hide_direct_parents: bez warunku 3 HCR czytalby
dokladnie te wartosci, ktore mial ukryc.

WSTAWIENIE. Zastap kazde wywolanie keep_observed(...) wywolaniem
keep_visible(...) w build_loaders w train_taskB.py - we WSZYSTKICH galeziach
(linear, nonlinear/binary, nonlinear/all, typed_concat).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import pandas as pd

# Re-eksport, NIE kopia. Slownik ma jedna definicje - w builderze. Kopiowanie
# go tutaj dalo by dwa zrodla prawdy: zmiana rezimu w jednym miejscu (np.
# usuniecie drug_exposure z "bedside") nie przenioslaby sie na drugie, a model
# liczylby sie na innym rezimie, niz wskazuje log.
from data.PreprocessingTaskB.build_patient_dag_heterodata_regime import (  # noqa: F401
    REGIME_VISIBLE_TYPES,
)

__all__ = ["keep_visible", "REGIME_VISIBLE_TYPES"]


def keep_visible(
    parents_by_endpoint: dict,
    samples_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    visible_types: Set[str],
    excluded_nodes: Optional[Set[str]] = None,
    verbose: bool = True,
    tag: str = "",
) -> dict:
    """Zostawia tylko rodzicow, ktorych model faktycznie moze zobaczyc.

    Obsluguje trzy ksztalty zwracane przez rozne gettery:
      get_binary_direct_parents       -> {ep: [nazwa, ...]}
      get_direct_parents_by_type      -> {ep: [(nazwa, typ), ...]}
      get_direct_parents_by_node_type -> {ep: {typ: [nazwa, ...]}}
    """
    observed = set(samples_df.columns)
    excluded = set(excluded_nodes or ())
    type_of = dict(zip(nodes_df["node"], nodes_df["node_type"]))

    def ok(p) -> bool:
        name = p[0] if isinstance(p, (tuple, list)) else p
        return (
            name in observed
            and name not in excluded
            and type_of.get(name) in visible_types
        )

    out, kept, dropped = {}, 0, 0
    for ep, parents in parents_by_endpoint.items():
        if isinstance(parents, dict):
            new = {t: [p for p in lst if ok(p)] for t, lst in parents.items()}
            kept += sum(len(v) for v in new.values())
            dropped += sum(len(v) for v in parents.values()) - sum(len(v) for v in new.values())
        else:
            new = [p for p in parents if ok(p)]
            kept += len(new)
            dropped += len(parents) - len(new)
        out[ep] = new

    if verbose:
        label = f" ({tag})" if tag else ""
        print(f"[HCR{label}] rodzice widoczni w rezimie {sorted(visible_types)}: "
              f"zostawiono {kept}, odrzucono {dropped}")
        empty = [ep for ep, p in out.items()
                 if (sum(len(v) for v in p.values()) if isinstance(p, dict) else len(p)) == 0]
        if empty:
            print(f"[HCR{label}] endpointy BEZ zadnego widocznego rodzica ({len(empty)}): {empty}")
            print(f"[HCR{label}] -> HCR nie wniesie dla nich nic; to nie jest blad, "
                  f"tylko wlasciwosc rezimu")
    return out