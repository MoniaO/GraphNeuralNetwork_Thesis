# Graph Neural Network Thesis — Task A

Repozytorium pracy dyplomowej poświęconej rekonstrukcji skierowanych krawędzi
grafu przyczynowego farmakoterapii. Główne zadanie to heterogeniczne link
prediction na syntetycznym zbiorze GSN v3, ze szczególnym uwzględnieniem:

- HGT i heterogenicznego message passingu;
- cech zależności HCR/GHCR;
- porównania decoderów MLP i KAN;
- odporności na sześć scenariuszy generowania pacjentów;
- jakości krawędzi należących do ścieżek prowadzących do rzadkich endpointów.

## Problem

Model otrzymuje dodatnie krawędzie splitu treningowego jako graf message
passingu. Następnie ocenia kandydackie pary węzłów ze splitów
train/validation/test. Dodatnie krawędzie validation i test pozostają ukryte
przed encoderem.

```text
pacjenci train ──► empiryczne cechy węzłów ─┐
                                             ├─► HGT ─► embeddingi węzłów ─┐
dodatnie krawędzie train ─► G_train ─────────┘                              │
                                                                            ├─► logit krawędzi
cechy GHCR par AZ, AG, ZG ─► MLP albo KAN ──────────────────────────────────┘
```

Wiersze pacjentów nie są przykładami treningowymi GNN. Służą do obliczenia
cech węzłów i dopasowania HCR/GHCR wyłącznie na partycji pacjentów `train`.

## Finalne modele

Oba modele mają identyczny:

- encoder HGT: 3 warstwy, hidden dimension 32, 8 głów attention, residual,
  LayerNorm i dropout 0.2;
- graf kandydatów, splity, seedy, loss i protokół ewaluacji;
- 40-wymiarowe wejścia GHCR dla trzech ról `AZ`, `AG`, `ZG`;
- graph branch i końcowy decoder.

Różni je wyłącznie encoder każdej pary GHCR:

- `TA_MLP`: `40 → 16 → 8`, GELU, LayerNorm, dropout 0.1;
- `TA_KAN`: bezpośredni odpowiednik KAN z bazą spline;
- audyt K1–K5: alternatywne architektury KAN, z najlepszym wariantem K1
  shallow.

Źródła konfiguracji:

- `configs/taskA/mlp_full_retrain.yaml`
- `configs/taskA/kan_full_retrain.yaml`
- `configs/taskA/kan_architectures/`
- `configs/model/TaskA_hgt_final.yaml`
- `configs/hcr/final_ghcr.yaml`

Nazwy `TaskA_hgt_wave7c` i `w7c_b2_audit` pozostają historycznymi źródłami
aliasów i służą do odtwarzania eksperymentów Wave 7C–10.

## Struktura repozytorium

```text
.
├── configs/                  konfiguracje Hydra
│   ├── data/                 GSN v3 i scenariusze
│   ├── model/                architektury GNN
│   ├── hcr/                  warianty HCR/GHCR
│   └── taskA/                finalny MLP, KAN i audit K1–K9
├── src/
│   ├── train_taskA.py        główny entrypoint treningu
│   ├── data/                 preprocessing i kandydaci krawędzi
│   ├── models/TaskA/         HGT, decodery i pair encoders
│   ├── hcr/                  implementacja HCR/GHCR
│   ├── evaluation/           metryki globalne i endpoint-path
│   ├── experiments/          wspólne funkcje audytów
│   └── wnerw/                historyczne eksperymenty ścieżkowe
├── scripts/                  runnery, ewaluacja i agregacja wyników
├── tests/                    testy jednostkowe i leakage guards
├── notebooks/               eksploracja danych i raporty wcześniejszych fal
└── outputs/
    └── Wave 0-11 podsumowanie/  wersjonowany raport promotorski
```

Szczegółowe mapy:

- `src/README.md` — przepływ kodu i kolejność czytania;
- `configs/README.md` — składanie konfiguracji;
- `scripts/README.md` — finalne i historyczne skrypty eksperymentalne;
- `outputs/Wave 0-11 podsumowanie/00_README.md` — historia i wyniki fal 0–11.

## Dane

Dane GSN v3 są przechowywane poza repozytorium. Ustaw:

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
```

Oczekiwana lokalizacja danych:

```text
$GSN_PROJECT_ROOT/
└── 2 v3. Data/
    ├── dataset_v3/
    │   ├── synthetic_pharmacotherapy_v3_nodes.csv
    │   ├── synthetic_pharmacotherapy_v3_edges_audited.csv
    │   └── synthetic_pharmacotherapy_v3_samples_<scenario>.csv
    └── splits/
        └── patient_splits_v3.csv
```

Scenariusze: `clean`, `hidden_confounder`, `selection_bias`, `no_overlap`,
`noisy_documentation` i `multihospital`.

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
```

W&B jest opcjonalne dla testów i lokalnych smoke runów. Pełne eksperymenty
korzystają z projektu `politechnika-gnn-thesis`.

## Szybka weryfikacja

Testy:

```bash
.venv/bin/python -m pytest
```

Sprawdzenie kompozycji finalnej konfiguracji bez treningu:

```bash
PYTHONPATH=src .venv/bin/python src/train_taskA.py \
  --cfg job \
  model=TaskA_hgt_final \
  hcr=final_ghcr \
  wandb.enabled=false
```

Smoke test encoderów:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_taskA_architecture.py
```

Jednoepokowy run kontrolny bez zapisu do W&B:

```bash
PYTHONPATH=src .venv/bin/python src/train_taskA.py \
  model=TaskA_hgt_final \
  hcr=final_ghcr \
  data.dataset.scenario=clean \
  data.candidate_seed=20260722 \
  training.seed=20260722 \
  training.epochs=1 \
  training.device=cpu \
  wandb.enabled=false
```

## Finalny eksperyment Wave 11

Najpierw zbuduj rejestr kontekstu krawędzi Wave 5C (wymaga evidence Wave 5 w
`outputs/wave5/evidence/`). Bez tego `hcr=final_ghcr` nie ma pełnych motywów
40D×3 i trening się wyłoży na brakującym pliku:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_wave5c_context_registry.py
```

Następnie:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_taska_mlp_vs_kan_scenarios.py
PYTHONPATH=src .venv/bin/python scripts/run_taska_kan_architecture_audit.py
PYTHONPATH=src .venv/bin/python scripts/run_taska_winner_rare_endpoint_audit.py
PYTHONPATH=src .venv/bin/python scripts/upload_taska_results_to_wandb.py
```

Ciężkie checkpointy, logi Hydra i lokalne runy W&B są ignorowane przez Git.
Wersjonowany jest jedynie lekki raport podsumowujący w
`outputs/Wave 0-11 podsumowanie/`.

## Metryki

Główna metryka selekcji modelu to validation AUPRC. Raport zawiera także
AUROC, Brier score, precision, recall, F1, F2, AUPRC lift, normalized AUPRC,
metryki oversmoothing oraz wyniki grupowane według ścieżek do endpointów.

Loss to ważona `BCEWithLogitsLoss`, gdzie `pos_weight` jest obliczany z bilansu
klas wyłącznie na zbiorze treningowym. Próg klasyfikacyjny jest wybierany na
validation i zamrażany dla testu.
