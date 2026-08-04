# Eksploracja grafu GSN v3

Ten katalog zawiera wyłącznie narzędzia diagnostyczne dla audytowanego grafu
GSN v3 używanego w Task A.

## Pliki

- `validate_v3.py` — walidacja schematu, typów, spójności referencji i
  acykliczności grafu.
- `visualize_v3.py` — interaktywna wizualizacja DAG w HTML.
- `analyze_graph_v3.ipynb` — eksploracja struktury i statystyk grafu.
- `v3_graph_diagnostics.ipynb` — raport diagnostyczny.
- `diagnostics_outputs/` — lekkie tabele wynikowe używane w notebookach.

## Dane

Skrypty korzystają z tej samej zmiennej środowiskowej co trening:

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
```

Oczekiwane pliki:

```text
$GSN_PROJECT_ROOT/2 v3. Data/dataset_v3/
├── synthetic_pharmacotherapy_v3_nodes.csv
├── synthetic_pharmacotherapy_v3_edges_audited.csv
└── synthetic_pharmacotherapy_v3_samples_<scenario>.csv
```

## Walidacja

Z katalogu głównego repozytorium:

```bash
.venv/bin/python "notebooks/Graph exploration/validate_v3.py"
```

Walidator sprawdza między innymi:

- obecność wymaganych kolumn;
- unikalność nazw węzłów;
- poprawność referencji `source` i `target`;
- brak self-loopów i duplikatów;
- zgodność typów oraz warstw;
- czy graf jest skierowanym grafem acyklicznym;
- czy endpointy są terminalne.

`PASS` oznacza zgodność techniczną z regułami generatora. Nie jest dowodem
klinicznej prawdziwości grafu ani wartości efektów.

## Wizualizacja

```bash
.venv/bin/python "notebooks/Graph exploration/visualize_v3.py"
```

Bez automatycznego otwierania przeglądarki:

```bash
.venv/bin/python "notebooks/Graph exploration/visualize_v3.py" \
  --no-open-browser
```

Skrypt zapisuje
`synthetic_pharmacotherapy_v3_dag.html` w katalogu danych GSN v3. Wizualizacja
pokazuje typ i warstwę węzła, kierunek relacji, typ krawędzi, znak i siłę
efektu, pathway, motyw DAG oraz rationale.
