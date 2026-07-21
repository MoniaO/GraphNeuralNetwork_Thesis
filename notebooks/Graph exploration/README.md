# Eksploracja, walidacja i wizualizacja grafu v3

Ten folder zawiera trzy niezależne narzędzia:

1. `validate_v3.py` — sprawdza techniczną spójność grafu i danych pacjentów;
2. `visualize_v3.py` — tworzy interaktywną wizualizację grafu w HTML;
3. `v3_graph_diagnostics.ipynb` — merytoryczna diagnostyka przed modelowaniem
   (scalony notebook; `analyze_graph_v3.ipynb` jest tylko przekierowaniem).

Walidacja nie zmienia danych. Wizualizacja i notebook również nie zmieniają
grafu — jedynie odczytują pliki `nodes.csv` i `edges.csv`.

---

## 1. Walidacja: `validate_v3.py`

### Uruchomienie

Z głównego folderu projektu:

```bash
".venv/bin/python" \
"3 v3. Graph exploration and diagnostics/validate_v3.py"
```

Raport jest zapisywany w:

```text
2 v3. Data/dataset_v3/v3_validation_report.json
```

Jeśli wszystkie obowiązkowe kontrole przejdą, raport zawiera:

```json
"status": "PASS",
"failures": []
```

W przypadku błędu otrzymuje `status: FAIL`, a program kończy się kodem błędu
`1`. Dzięki temu walidator może być później używany również w GitHub Actions.

### Co jest walidowane?

#### A. Struktura grafu

- liczba węzłów;
- liczba krawędzi;
- czy graf jest skierowany i acykliczny (`is_dag`);
- zduplikowane nazwy węzłów;
- zduplikowane pary `source → target`;
- krawędzie wskazujące na niezadeklarowane węzły.

Sprawdzenie DAG jest wykonywane przez:

```python
nx.is_directed_acyclic_graph(graph)
```

#### B. Kierunki krawędzi dla węzłów

Węzły są dzielone na:

- izolowane: brak krawędzi wchodzących i wychodzących (`0/0`);
- źródłowe: brak wchodzących, ale mają wychodzące (`0/>0`);
- końcowe: mają wchodzące, ale brak wychodzących (`>0/0`);
- pośrednie: mają krawędzie w obu kierunkach (`>0/>0`).

Znany izolowany węzeł:

```text
cognitive_slowing
```

Jest zapisany jako oczekiwany wyjątek. Każdy nowy, nieoczekiwany izolowany
węzeł powoduje `FAIL`. Węzły źródłowe i końcowe są raportowane informacyjnie,
ponieważ są prawidłowe w DAG.

#### C. Liczba aktywnych leków

Dla każdego pacjenta walidator ponownie sumuje wszystkie 29 zmiennych lekowych
i porównuje wynik z `active_drug_count`.

Raportuje również:

- średnią liczbę aktywnych leków;
- 5. percentyl;
- 95. percentyl.

Średnia musi znajdować się między 5,5 a 6,5.

#### D. Wielolekowe obciążenia (`drug loads`)

Walidator ponownie oblicza i porównuje:

- `nephrotoxin_load`;
- `hepatic_drug_load`;
- `qt_drug_load_v3`;
- `cns_depressant_load`;
- `bleeding_risk_load`;
- `serotonergic_load`;
- `cyp_inhibitor_load`.

Dla każdego load sprawdzana jest zgodność dla wszystkich pacjentów oraz liczba
pacjentów z wartością większą od zera.

#### E. Bramki interakcji

Ponownie obliczane są wszystkie bramki lek–lek i lek–choroba.

Przykład:

```text
drug_disease_nsaid_ckd = nsaid AND ckd
```

Dla `ddi_serotonergic_high_load` obowiązuje:

```text
serotonergic_load >= 2
```

Wartość każdej bramki musi dokładnie odpowiadać jej definicji.

#### F. Skumulowane obciążenie ADR

Walidator ponownie liczy:

```text
cumulative_adr_burden =
sum(komponent ADR × jego waga)
+ 0.08 × active_drug_count
```

Sprawdza też:

```text
severe_adr_burden = 1,
gdy cumulative_adr_burden >= 4
```

#### G. Scenariusze pacjentów

Kontrolowane są:

- `hidden_confounder` — brak `unobserved_severity` i te same identyfikatory co
  w `clean`;
- `selection_bias` — pacjenci są podzbiorem pacjentów z `clean`;
- `noisy_documentation` — istnieją kolumny `recorded_<endpoint>`;
- `multihospital` — występują dokładnie cztery szpitale.

Uwaga: aktualny walidator nie sprawdza jeszcze formalnie siły konstrukcji
`no_overlap`. Scenariusz został empirycznie sprawdzony osobno, ale nie jest
jeszcze częścią automatycznego statusu `PASS/FAIL`.

#### H. Częstości endpointów

Dla wszystkich endpointów sprawdzane jest, czy:

```text
0 < liczba pozytywnych przypadków < liczba pacjentów
```

Dodatkowo szerokie zakresy projektowe obowiązują dla:

- AKI;
- DILI;
- Falls;
- QT arrhythmia;
- Hospitalization;
- Serotonin syndrome;
- Rhabdomyolysis;
- Lactic acidosis.

Zakresy są założeniami symulacji, a nie klinicznymi normami częstości.

### Czego `PASS` nie oznacza?

`PASS` oznacza techniczną zgodność danych z zaprogramowanymi regułami.
Nie potwierdza:

- klinicznej prawdziwości grafu;
- poprawności klinicznej wartości `effect_size`;
- jakości rationale;
- jakości modelu GNN;
- inwariantności modelu;
- relacji przyczynowej w danych rzeczywistych.

---

## 2. Wizualizacja: `visualize_v3.py`

### Uruchomienie

```bash
"/Users/martajasiewicz/Desktop/synthetic_pharmacotherapy_v2/.venv/bin/python" \
"3 v3. Graph exploration and diagnostics/visualize_v3.py"
```

Wymagane jest środowisko zawierające `pyvis`.

Wynik:

```text
2 v3. Data/dataset_v3/synthetic_pharmacotherapy_v3_dag.html
```

Skrypt domyślnie próbuje automatycznie otworzyć HTML w przeglądarce. Aby tylko
utworzyć plik:

```bash
"/Users/martajasiewicz/Desktop/synthetic_pharmacotherapy_v2/.venv/bin/python" \
"3 v3. Graph exploration and diagnostics/visualize_v3.py" \
--no-open-browser
```

### Co pokazuje wizualizacja?

- 155 węzłów;
- 405 skierowanych krawędzi;
- kierunek `source → target`;
- typ i warstwę węzła;
- liczbę rodziców i dzieci;
- typ krawędzi;
- znak i siłę efektu;
- pathway;
- motyw DAG;
- rationale;
- wersję, w której dodano relację.

### Kolory węzłów

- szary — `patient_context`;
- niebieski — `drug_exposure`;
- fioletowy — `mechanism`;
- pomarańczowy — `adr_or_intermediate_state`;
- turkusowy — `observation_or_selection`;
- czerwony — `clinical_endpoint`.

### Jak eksplorować?

1. Wybierz węzeł z menu wyszukiwania.
2. Najedź na węzeł, aby zobaczyć rodziców, dzieci i opis.
3. Najedź na krawędź, aby zobaczyć typ, efekt, pathway i rationale.
4. Użyj filtrów `node_type`, `layer`, `edge_type` lub `clinical_pathway`.
5. Użyj `Reset Selection`, aby wrócić do pełnego grafu.
6. Ustawienia `physics` pozwalają zmienić rozmieszczenie węzłów.

### Czego wizualizacja nie pokazuje?

HTML przedstawia globalny graf wiedzy. Nie przedstawia:

- 20 000 indywidualnych pacjentów;
- osobnego grafu dla każdego pacjenta;
- aktywacji ścieżki u konkretnego pacjenta;
- wyników predykcji GNN;
- różnic częstości pomiędzy scenariuszami.

Scenariusze pacjentów korzystają z tej samej topologii, ale różnią się
obserwowanymi danymi.

---

## 3. Diagnostyka merytoryczna: `v3_graph_diagnostics.ipynb`

Notebook do eksploracji przed GNN (połączenie dawnego `analyze` + `diagnostics`). Zawiera:

- integralność DAG, warstwy i typy węzłów;
- stopnie, huby, źródła, ujścia, izolaty;
- głębokość topologiczną (warstwy przyczynowe);
- typy krawędzi i clinical pathways;
- najkrótsze i najdłuższe ścieżki do endpointów;
- motywy przyczynowe: mediacja, fork, collider;
- przodków i potomków endpointów;
- centralności globalne (betweenness, PageRank);
- confounding / backdoor: wspólni przodkowie lek→endpoint;
- ranking strukturalny leków (out-degree, weighted out-degree,
  reachable ADR/endpoints, PageRank, betweenness, Katz, harmonic closeness);
- korelacje tych miar z syntetycznymi proxy ADR w `clean`
  (diagnostyka spójności projektu, nie walidacja kliniczna);
- checklistę gotowości do modelowania.

Nie zawiera liczby receptorów/narządów — v3 nie ma warstwy molekularnej.

Wyniki tabelaryczne zapisuje w:

```text
3 v3. Graph exploration and diagnostics/diagnostics_outputs/
```

W tym m.in.:

- `dag_integrity_summary.json` — rozmiar, DAG, spójność;
- `topological_depth.csv` — głębokość przyczynowa węzłów;
- `endpoint_ancestors_descendants.csv` — przodkowie/potomkowie endpointów;
- `node_centralities.csv` — betweenness / PageRank całego grafu;
- `drug_endpoint_common_ancestors.csv` — kandydaci confounding / backdoor;
- `drug_structural_ranking.csv` — ranking leków + proxy z `clean`;
- `drug_metric_vs_synthetic_adr_correlations.csv` — Spearman metryka ↔ proxy ADR;
- `modeling_readiness_checklist.json` — checklista przed GNN.

Uruchom notebook w Jupyter / VS Code / Cursor z katalogiem roboczym:

```text
3 v3. Graph exploration and diagnostics/
```
