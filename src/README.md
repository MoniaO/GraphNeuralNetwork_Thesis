# Task B — Predykcja endpointów klinicznych na grafie przyczynowym

---

## 1. Co robi Task B (w skrócie)

Mamy **znaną prawdę przyczynową** — audytowany DAG (`edges_audited.csv`) —
identyczną dla Task A i Task B. W przeciwieństwie do Task A, tutaj **pacjent
jest jednostką treningową**: każdemu pacjentowi odpowiada jeden
`HeteroData`, ze **stałą** topologią krawędzi (ta sama struktura i wagi dla
każdego pacjenta) i **zmienną** realizacją wartości węzłów (`x`).

Model dostaje częściowo obserwowalny stan pacjenta — który dokładnie
podzbiór węzłów jest widoczny, kontroluje **reżim obserwowalności**
(`full` / `mechanisms_latent` / `bedside`) — i musi przewidzieć etykiety na
węzłach typu `clinical_endpoint` (AKI, Hospitalization, Depression, …).

```text
audytowany DAG (nodes.csv + edges_audited.csv)
        │  (stała topologia + wagi, identyczna dla kazdego pacjenta)
        ▼
per-pacjent realizacja wartości węzłów (samples_<scenario>.csv)
        │
        ├─ standaryzacja x[:,0] (mean/std liczone WYŁĄCZNIE na train)
        ├─ maskowanie wg observability_regime (kanał is_observed)
        └─ wykluczenie leakage: unobserved_severity (is_latent) +
           strukturalni potomkowie endpointów (hospital_contact,
           fall_reported, adr_reported, mood_screening_done)
        │
        ▼
HeteroData per pacjent
  (x_dict + wspólny edge_index/edge_attr + y na węzłach clinical_endpoint)
        │
        ▼
encoder GNN: HeteroConv(sage/gcn/gat/transformer)  LUB  R-GCN
             (spłaszczona przestrzeń węzłów, pełna granulacja relacji,
              dekompozycja bazowa)
        │
        ▼
NodeClassificationHead → logity dla wybranej podlisty endpointów
        │
        ▼
AUC/AUPRC (macro) + oversmoothing (relation_macro/edge_weighted,
MAD, cosine, numerical_rank per warstwa) → W&B + checkpoint
```

---

## 2. Dane — skąd i jakie pliki

| Plik | Rola w Task B |
|------|------|
| `synthetic_pharmacotherapy_v3_nodes.csv` | węzły grafu (nazwa, typ, warstwa, `recommended_use`, `is_latent`, `is_endpoint`, …) |
| `synthetic_pharmacotherapy_v3_edges_audited.csv` | pełny `G_true` — **stała** topologia message-passingu dla każdego pacjenta |
| `synthetic_pharmacotherapy_v3_samples_<scenario>.csv` | pacjenci — **wiersze treningowe** (jeden HeteroData per pacjent) |
| `splits/patient_splits_v3.csv` | **faktyczny** podział train/valid/test pacjentów (nie tylko do liczenia statystyk jak w Task A) — wielo-etykietowo zbalansowany, dziedziczony przez wszystkie scenariusze z `clean` |

### Scenariusze pacjentów

Te same sześć co w Task A (`clean`, `hidden_confounder`, `selection_bias`,
`no_overlap`, `noisy_documentation`, `multihospital`) — ten sam `G_true`,
różne światy obserwacji. 

**Aktualny baseline:** `scenario: clean`, `observability_regime: full`.

---

## 3. Jak powstaje wejście do modelu

### 3.1 Cechy węzłów

W przeciwieństwie do Task A (agregaty populacyjne: `structural` /
`empirical` / `oracle`), Task B używa **realizacji per pacjent**:

| Składowa `x` węzła | Opis |
|---|---|
| `x[:,0]` — realizacja | Wartość zmiennej u TEGO pacjenta, standaryzowana (mean/std z **train**) |
| `x[:,1]` — `is_observed` | 0/1: czy wartość jest widoczna w bieżącym reżimie obserwowalności |
| cechy statyczne | `severity_weight`, `observability`, `rarity_weight`, `node_priority_weight`, `layer_numeric` (z `nodes.csv`, fillna medianą, nie zerem) |
| embedding tożsamości | `nn.Embedding` per typ węzła, indeksowany `node_idx` — rozwiązuje kolizje węzłów o identycznych cechach statycznych |

**Reżimy obserwowalności** (`REGIME_VISIBLE_TYPES`, brak odpowiednika w
Task A):

| Reżim | Widoczne typy węzłów |
|---|---|
| `full` | wszystko poza wykluczonymi (137/155 kolumn) |
| `mechanisms_latent` | bez typu `mechanism` (88/155) — sygnał musi płynąć 3–5 hopów |
| `bedside` | tylko `patient_context` + `drug_exposure` (52/155) — to, co lekarz ma przy przyjęciu |

**Wykluczenia (leakage), wykrywane strukturalnie, nie ręcznie:**
- `unobserved_severity` — `is_latent=True`, ukryty konfounder.
- `hospital_contact`, `fall_reported`, `adr_reported`, `mood_screening_done`
  — **potomkowie endpointów** w DAG-u (np. `Hospitalization → hospital_contact`,
  `effect_size=1.0`)

### 3.2 Etykiety i split

W Task B jednostką modelowaną jest **pacjent**. Split
70/15/15, deterministyczny (hash), **wielo-etykietowo zbalansowany**
(greedy, respektujący rozkład pozytywów per endpoint w każdym splicie),
liczony raz na `clean` i dziedziczony przez wszystkie scenariusze.

### 3.3 Model

Kod: `models/TaskB/gnn_node.py` (HeteroConv), `models/TaskB/gnn_node_rgcn.py`
(R-GCN), `models/TaskB/gnn_common.py` (współdzielone: głowica, etykiety).

| Komponent | Opis |
|---|---|
| `TargetedPatientDAGNodeClassifier` | ogranicza predykcję do wybranej podlisty (domyślnie 10); kilka mozliwosci konwolucji |
| `RGCNPatientDAGNodeClassifier` | R-GCN na spłaszczonej przestrzeni węzłów, pełna granulacja relacji (nie 19 jak HeteroConv) dzięki dekompozycji bazowej — osobny plik, bo mechanika (`x_dict`/`edge_index_dict` vs jeden globalny tensor) jest fundamentalnie inna |

Model **w pełni wykorzystuje** `edge_attr`,
nie tylko typ relacji w kluczu heterogenicznym.

`conv_type ∈ {sage, gcn, gat, transformer}` przez `HeteroConv`; `rgcn`
osobną ścieżką. 

---

## 4. Trening i wyniki

Wejście: `train_taskB.py` (Hydra + W&B).

### Co optymalizujemy

- Loss: `BCEWithLogitsLoss` z `pos_weight` per endpoint (z bilansu klas na
  train, przycięty do max 30).
- Selekcja checkpointu: `valid/auprc`.
- Próg klasyfikacji (`f1`/`precision`/`recall`): wyznaczany **w tym samym
  forward-passie** co `valid_metrics` (`select_threshold=True`), świeży co
  epokę, bez podwajania kosztu obliczeniowego. Finalna ewaluacja po
  `model.load_state_dict(best_state)` przelicza próg jeszcze raz, na
  najlepszym checkpoincie — to liczby do cytowania w pracy, nie logi
  epokowe.

### Tagi/grupy W&B

`conv_type`, `regime` (obserwowalności), `layers`, `residual`, `scenario`,
`model`, `targets` — żeby dało się filtrować/grupować po każdym wymiarze
eksperymentu osobno.

---

## 5. Metryki — co i gdzie liczymy

Implementacja: `evaluation/syntetic_evaluator_node.py` (+ **wspólne** z
Task B: `evaluation/oversmoothing_metrics.py`

| Metryka | Uwagi specyficzne dla Task B |
|---|---|
| **AUC / AUPRC** | macro po endpointach (sklearn default dla `multilabel-indicator`); rzadkie endpointy (mało pozytywów w walidacji) dominują wariancję macro — stąd nieregularne wykresy `valid/auc` mimo gładkiego `valid/auprc` |
| **Bayes ceiling** (planowane) | dzięki dostępowi do generatora można policzyć `AUC_oracle` z prawdziwego `eta`/`sigmoid` i raportować `(AUC−0.5)/(AUC_oracle−0.5)` — mocny argument metodologiczny, jeszcze nie zaimplementowany jako kod |
| **oversmoothing/** | `cosine_sim` / `mad` / `numerical_rank` (mianownik `min(n-1,dim)`) / `feature_std` per typ węzła; `dirichlet_energy_relation_macro` (każda relacja równa) i `dirichlet_energy_edge_weighted` (każda krawędź równa, micro) — **ten sam schemat co w Task A**, zsynchronizowane z koleżanką |
| **f1/precision/recall** | `average='macro'` ma różne znaczenie w zależności od kształtu danych (agregat vs pojedynczy endpoint) — znane, jeszcze nienaprawione zniekształcenie przy `eval_per_endpoint=True` |

---

## 6. Struktura repozytorium 

```text
GraphNeuralNetwork_Thesis/
├── configs/
│   ├── model/TaskB_hetero_sage.yaml, TaskB_rgcn.yaml, ...
│   └── data/dataset_v3.yaml            # WSPÓLNY z Task A
├── src/
│   ├── train_taskB.py
│   ├── data/PreprocessingTaskB/
│   │   └── build_patient_dag_heterodata.py
│   ├── models/TaskB/
│   │   ├── gnn_common.py               # głowica, etykiety - WSPÓLNE dla sage/gcn/gat/transformer/rgcn
│   │   ├── gnn_node.py                 # sage/gcn/gat/transformer (HeteroConv)
│   │   └── gnn_node_rgcn.py            # R-GCN (spłaszczona przestrzeń)
│   └── evaluation/
│       ├── syntetic_evaluator_node.py
│       └── oversmoothing_metrics.py    # WSPÓLNY z Task A - koordynować zmiany
```

---

## 7. Plan uczenia dalej

### Faza 0 — baseline na `clean`, reżim `full`

- Architektury: sage, gcn, gat, transformer, rgcn.
- Głębokość 1–6 warstw × `use_residual` on/off 
- Przyjrzenie się `root_weight` dla sage/transformer 

### Faza 1 — reżimy obserwowalności (główna oś eksperymentu, unikalna dla Task B)

`full` vs `mechanisms_latent` vs `bedside` — maskowanie zmiennych w zaleznosci od tego jak moga wplywac na predykcje

### Faza 2 — ablacje struktury

MLP bez grafu / graf z permutowanymi krawędziami / graf pełny / prawdziwy
DAG — potwierdzenie, że to **struktura przyczynowa**, nie sam fakt bycia
GNN-em, niesie wartość predykcyjną.

### Faza 3 — porównanie scenariuszy (ten sam protokół co Task A Faza 1)

Ten sam model/hiperparametry na 6 scenariuszach, `test/auprc` + lift
względem `clean` w W&B.

---

## 8. Najważniejsze zasady

1. **Pacjent = jednostka treningowa**, nie źródło statystyk (różnica od Task A).
2. **Standaryzacja i podział train/valid/test liczone wyłącznie na train**, przed zbudowaniem grafów.
4. Reżim obserwowalności jest głównym pokrętłem eksperymentu o oversmoothing — głębsze warstwy mają znaczenie tylko, gdy sygnał faktycznie musi przez nie płynąć.
