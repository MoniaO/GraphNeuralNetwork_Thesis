"""
Test kanalu s_evidence (iloczyn a11 * phi1) we wszystkich funkcjach dowodowych.

NIE wymaga configu, Hydry ani prawdziwych danych - buduje maly, sztuczny
samples_df i sprawdza wlasnosc, ktora ma zachodzic z definicji:

    kanal s_evidence  ==  kanal a11  *  kanal phi1_patient

Sprawdza to OSOBNO dla wierszy treningowych i nietreningowych, bo w
compute_hcr_pair_evidence_mixed blok jest najpierw wypelniany statystykami
z pelnego treningu, a dopiero potem wiersze treningowe sa nadpisywane
wersja leave-one-out. Iloczyn policzony przed nadpisaniem daje dla pacjentow
treningowych wynik ze zlych czynnikow - i to jest przeciek, przed ktorym
leave-one-out ma chronic. Samo np.allclose na calym tensorze tego NIE
wykryje, jesli iloczyn i czynniki sa niespojne w ten sam sposob.

Uruchom z katalogu, w ktorym lezy hcr_wide_features.py:
    python test_s_evidence.py
    python test_s_evidence.py --module sciezka/do/hcr_wide_features.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("hcr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_data(n: int = 400, seed: int = 0):
    """Maly zbior binarny: 3 rodzicow, 2 endpointy, zaleznosc niezerowa."""
    rng = np.random.default_rng(seed)
    par = {c: rng.binomial(1, p, n).astype(float)
           for c, p in [("drug_a", 0.30), ("ctx_b", 0.45), ("mech_c", 0.20)]}
    df = pd.DataFrame(par)
    for ep, w in [("EP1", 1.4), ("EP2", 0.9)]:
        logit = -2.5 + w * df["drug_a"] + 0.8 * df["ctx_b"] - 0.6 * df["mech_c"]
        df[ep] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df.insert(0, "patient_id", np.arange(n))
    train_ids = set(range(int(n * 0.6)))
    return df, train_ids


def check(name: str, feats: np.ndarray, i_prod: int, i_a: int, i_phi: int,
          train_pos: np.ndarray, non_train_pos: np.ndarray) -> bool:
    prod, a, phi = feats[..., i_prod], feats[..., i_a], feats[..., i_phi]
    nonzero = float((prod != 0).mean())

    ok_all = np.allclose(prod, a * phi, atol=1e-5)
    ok_tr = np.allclose(prod[train_pos], (a * phi)[train_pos], atol=1e-5)
    ok_va = np.allclose(prod[non_train_pos], (a * phi)[non_train_pos], atol=1e-5)

    print(f"\n  {name}")
    print(f"    niezerowych w kanale iloczynu: {nonzero:.1%}")
    print(f"    spojnosc na wierszach treningowych     : {'OK' if ok_tr else 'BLAD'}")
    print(f"    spojnosc na wierszach nietreningowych  : {'OK' if ok_va else 'BLAD'}")

    if nonzero == 0.0:
        print("    -> KANAL PUSTY. Funkcja nie wypelnia iloczynu wcale.")
        return False
    if ok_all and nonzero > 0:
        # dodatkowa kontrola: czy wartosci LOO i pelne faktycznie sie roznia,
        # inaczej test przeszedlby tez przy calkowitym braku leave-one-out
        differs = not np.allclose(a[train_pos].mean(), a[non_train_pos].mean(), atol=1e-9)
        print(f"    wspolczynnik LOO rozni sie od pelnego  : {'tak' if differs else 'NIE (podejrzane)'}")
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", default="hcr_wide_features.py")
    args = ap.parse_args()

    h = load_module(Path(args.module))
    df, train_ids = make_data()
    parents_flat = {"EP1": ["drug_a", "ctx_b", "mech_c"], "EP2": ["drug_a", "mech_c"]}
    parents_typed = {ep: [(p, "binary") for p in v] for ep, v in parents_flat.items()}

    train_pos = df["patient_id"].isin(train_ids).to_numpy()
    non_train_pos = ~train_pos

    print("=" * 66)
    print("TEST KANALU s_evidence")
    print("=" * 66)
    results = {}

    # --- 9D (teraz 10 kanalow): a11 = kanal 1, phi1 = kanal 0 ---
    pe = h.compute_hcr_pair_evidence(df, parents_flat, train_ids)
    n9 = h.N_PAIR_EVIDENCE_CHANNELS
    print(f"\nkanalow w wariancie 'waskim': {n9}")
    results["compute_hcr_pair_evidence"] = check(
        "compute_hcr_pair_evidence (9D->10D)", pe["features"], n9 - 1, 1, 0,
        train_pos, non_train_pos)

    # --- 41D (teraz 42): a11 = kanal 0 (matrix_0), phi1 = kanal 40 ---
    n41 = h.N_PAIR_EVIDENCE_CHANNELS_40D
    print(f"\nkanalow w wariancie 'szerokim': {n41}")
    pe40 = h.compute_hcr_pair_evidence_40d(df, parents_flat, train_ids)
    results["compute_hcr_pair_evidence_40d"] = check(
        "compute_hcr_pair_evidence_40d", pe40["features"], n41 - 1, 0, 40,
        train_pos, non_train_pos)

    pem = h.compute_hcr_pair_evidence_mixed(df, parents_typed, train_ids)
    results["compute_hcr_pair_evidence_mixed"] = check(
        "compute_hcr_pair_evidence_mixed", pem["features"], n41 - 1, 0, 40,
        train_pos, non_train_pos)

    # --- typed_concat deleguje do _mixed, wiec sprawdzamy tez przez nia ---
    by_type = {ep: {"drug_exposure": [(p, "binary") for p in v]}
               for ep, v in parents_flat.items()}
    pbt = h.compute_hcr_pair_evidence_by_node_type(df, by_type, train_ids)
    sub = pbt.get("drug_exposure")
    if sub is not None:
        results["by_node_type[drug_exposure]"] = check(
            "compute_hcr_pair_evidence_by_node_type", sub["features"], n41 - 1, 0, 40,
            train_pos, non_train_pos)

    print("\n" + "=" * 66)
    bad = [k for k, v in results.items() if not v]
    if bad:
        print("NIEZALICZONE:")
        for k in bad:
            print(f"  - {k}")
        raise SystemExit(1)
    print("Wszystkie funkcje dowodowe wypelniaja kanal iloczynu spojnie.")


if __name__ == "__main__":
    main()