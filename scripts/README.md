# Skrypty eksperymentalne Task A

Wszystkie skrypty w tym katalogu dotyczą rekonstrukcji krawędzi (link
prediction). Kod modelu znajduje się w `src/`, a skrypty jedynie uruchamiają,
audytują lub agregują eksperymenty.

## FINAL 14.08.2026 (aktualny stack — HGT L2 + fusion88_stat + S10)

Pełna mapa zależności: `outputs/taskA_FINAL_14.08.2026/CODE_AND_SCRIPT_MAP_14.08.2026.md`

1. `run_taskA_FINAL_14.08.2026.py` — grid 60 jobów (S10 MLP + KAN × 6 × 5)
2. `watch_FINAL_14.08.2026.py` — watchdog
3. `plot_taska_learning_curves_14.08.2026.py` — krzywe train/valid
4. `eval_taska_stage_c_edge_pathway_report.py` — edge × prob × clinical_pathway / rare

Poprzednia kampania freeze (11.08): `run_taskA_stage_a_backbone_11.08.2026.py`,
`run_taskA_stage_c_stats_11.08.2026.py`.

## Finalny eksperyment — Wave 11

Uruchamiaj w tej kolejności:

0. `build_wave5c_context_registry.py` — rejestr kontekstu krawędzi
   (`outputs/wave5c/registry/edge_context_registry.csv`). Wymaga evidence
   Wave 5 w `outputs/wave5/evidence/`. Bez tego attach Wave 7C/11 nie ma
   pełnych motywów AZ/AG/ZG.
1. `run_taska_mlp_vs_kan_scenarios.py` — pełny retrain finalnego MLP i
   bezpośredniego KAN: 6 scenariuszy × 3 seedy.
2. `build_taska_edge_endpoint_registry.py` — rejestr ścieżek prowadzących do
   endpointów.
3. `eval_taska_run_endpoint_paths.py` — metryki endpoint-path dla pojedynczego
   runu.
4. `run_taska_kan_architecture_audit.py` — screening architektur K1–K5.
5. `run_taska_kan_arch_multiseed.py` i
   `run_taska_kan_arch_full_scenarios.py` — walidacja wybranego KAN.
6. `run_taska_winner_rare_endpoint_audit.py` — wspólna tabela MLP–KAN dla
   rzadkich endpointów.
7. `upload_taska_results_to_wandb.py` — wysłanie zagregowanych tabel do W&B.

Konfiguracje finalnych modeli:

- `configs/model/TaskA_hgt_final.yaml`
- `configs/hcr/final_ghcr.yaml`
- `configs/taskA/mlp_full_retrain.yaml`
- `configs/taskA/kan_full_retrain.yaml`
- `configs/taskA/kan_architectures/`

## Task A — Proxy Validation na JODIE MOOC

Główny runner `run_mooc_edge_emergence_proxy.py` zachowuje binarny cel Task A.
Z grafu student–aktywność buduje historyczny graf skierowanych przejść między
97 aktywnościami. Kandydatem jest niewidziana dotąd para `A→G`, a etykieta
określa, czy krawędź pojawi się w następnym oknie:

- 0–70% zdarzeń buduje graf historii,
- 70–80% służy jako przyszły horyzont treningowy,
- 80–90% jest walidacją,
- 90–100% jest testem użytym dopiero po wyborze epoki.

Self-loopy są wykluczone. Registry ma zamrożony stosunek 1:3 oraz dopasowane
negatywy, więc bazowe AUPRC wynosi 0.25. Główna macierz populacyjna zawiera
historyczne liczby interakcji `student × aktywność`; binary HCR jest kontrolą.
Porównywane są heurystyki, klasyczne zależności, HCR-only, HGT-only,
HGT+classical, HGT+HCR, shuffled/permuted HCR, AG-only i shared encoder.
Raport obejmuje także pełny universe kandydatów z naturalną częstością klas.

MOOC sprawdza przenoszalność reprezentacji zależności HCR, ale bez danych
interwencyjnych nie dowodzi przyczynowości. `data.y` nie jest wejściem modelu.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_edge_emergence_proxy.py
```

Szybki smoke test:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_edge_emergence_proxy.py \
  --smoke --models hgt_only hgt_hcr \
  --output-dir outputs/taskA_proxy_validation/mooc_edge_emergence_smoke
```

Historyczny `run_mooc_taska_hcr_proxy.py` rozwiązuje inne zadanie next-item
(1 z 97 klas) i pozostaje wyłącznie analizą pomocniczą, nie walidacją Task A.

Audyt temporalnego MOOC-B:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_edge_control_audit.py
PYTHONPATH=src .venv/bin/python scripts/run_mooc_edge_jitter_audit.py
```

Główna walidacja strukturalna MOOC-A rozdziela studentów na trzy niezależne
kohorty: budującą topologię, dostarczającą HCR i definiującą etykiety
brakujących krawędzi. Trzy rotacje zamieniają role kohort:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_cross_cohort.py
```

Wariant główny używa deterministycznego `count_interval_mean` HCR, który
całkuje bazę Legendre'a po przedziale randomized PIT zamiast wybierać jeden
losowy jitter.

Rozszerzenie higher-order korzysta wyłącznie z zamrożonego
`MOOC_A_V1_FROZEN`: dodaje osobno binary CMI 6D i interval-mean triple HCR
10D, zachowując te same kohorty, kandydatów, negatywy i kontekst Z:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_cross_cohort_higher_order.py
```

Osobny benchmark neuronowy porównuje G0 oraz HGT R0–R5 w pięciu sparowanych
seedach. Nie należy utożsamiać go z bazowym wynikiem `0.403`, który pochodzi
z deterministycznej regresji logistycznej:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_cross_cohort_neural.py
```

Po zamrożeniu decyzji reprezentacyjnej porównanie architektur zmienia
wyłącznie encoder topologii. `S0/S1` oznaczają Directed GraphSAGE bez/z HCR,
a `N0/N1` — skierowany NCN-style encoder pary bez/z HCR. Kontrola shuffled
uruchamia się automatycznie tylko dla rodziny spełniającej zamrożony gate:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_architecture_comparison.py \
  --auto-shuffled
```

Manifest `MOOC_A_REPRESENTATION_AUDIT_V1` zachowuje CMI i triple-HCR jako
`exploratory_only`; porównanie architektur używa wyłącznie pairwise HCR.

Po odrzuceniu concat HCR przy NCN (`N1`), kolejny eksperyment testuje HCR jako
bramkę wagującą ścieżki `A→Z→G` (`MOOC_A_HCR_GATED_NCN_V1`). Najpierw zbuduj
rejestr kontekstów i cache HCR, potem uruchom N2/N3:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_mooc_context_registry.py \
  --manifest MOOC_A_V1_FROZEN
PYTHONPATH=src .venv/bin/python scripts/build_mooc_context_hcr.py \
  --count-method interval_mean
PYTHONPATH=src .venv/bin/python scripts/run_mooc_hcr_gated_ncn.py \
  --variants N2 N3 \
  --rotations 0 1 2 \
  --seeds 0 1 2 3 4 \
  --auto-controls
```

`--auto-controls` uruchamia N5/N6 tylko gdy N3 wygrywa z N2 według zamrożonego
gate (≥10/15 sparowanych runów i pozostałe warunki z protokołu).

Segment-conditioned HCR (`MOOC_A_SEGMENT_CONDITIONED_HCR_V1`) nie zmienia wzoru
HCR — zmienia populację referencyjną. Kandydaci i negatywy zostają z
`MOOC_A_V1_FROZEN`. Segmentacja jest fitowana wyłącznie na studentach kohorty
HCR; `k` nie wolno wybierać z testowego AUPRC NCN+HCR.

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_mooc_sc_hcr_manifest.py
PYTHONPATH=src .venv/bin/python scripts/run_mooc_sc_hcr_stage_a.py
PYTHONPATH=src .venv/bin/python scripts/run_mooc_sc_hcr_stage_b.py
PYTHONPATH=src .venv/bin/python scripts/run_mooc_sc_hcr_stage_c.py \
  --variants C0 C1 C2 C3 C4 C5_VECTOR C6_VECTOR_GATE --auto-controls
```

Na MOOC-A wynik Stage B to *cohort-/regime-conditioned* HCR (nie student-conditioned).
C5/C6 podają pełny wektor `[h_G,h_1..h_K]` zamiast tylko uśrednienia przez stałe `π_k`.
Primary gate: `C6_VECTOR_GATE` vs `C0`.

Po Stage C (gate fail) zamroź decyzję i uruchom Stage D — rezydualizacja HCR:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_mooc_sc_hcr_decision.py
PYTHONPATH=src .venv/bin/python scripts/run_mooc_sc_hcr_stage_d.py \
  --variants D0 D1 D2 D3 D4
```

Temporalne MOOC-B z prawdziwym student-conditioned SC-HCR (`q(k|u,t)`):

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_mooc_b_sc_hcr_manifest.py
PYTHONPATH=src .venv/bin/python scripts/build_mooc_b_student_registry.py
PYTHONPATH=src .venv/bin/python scripts/run_mooc_b_sc_hcr_stage_a.py
PYTHONPATH=src .venv/bin/python scripts/run_mooc_b_sc_hcr_stage_b.py
# Stage C tylko po przejściu bramki Stage B:
PYTHONPATH=src .venv/bin/python scripts/run_mooc_b_sc_hcr_stage_c.py
```

Smoke (1 segmenter seed, 1 model seed, 1 epoka):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_b_sc_hcr_stage_b.py --smoke
PYTHONPATH=src .venv/bin/python scripts/run_mooc_b_sc_hcr_stage_c.py \
  --smoke --allow-failed-stage-b
```

Kandydaci to `(student, A→G)` w oknie etykiet, nie populacyjne pary bez `u`.
Wyniki: `outputs/taskA_proxy_validation/MOOC_B_STUDENT_CONDITIONED_HCR_V1/`.

Po Stage B zamroź decyzję (pełne Stage C tylko gdy bramka B przejdzie):

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_mooc_b_sc_hcr_decision.py
```

MOOC-B V2 (benchmark tylko `context_mask==1`, bez confounded shift):

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_mooc_b_sc_hcr_v2_manifest.py
PYTHONPATH=src .venv/bin/python scripts/build_mooc_b_student_registry_v2.py
PYTHONPATH=src .venv/bin/python scripts/run_mooc_b_sc_hcr_stage_a.py \
  --output-dir outputs/taskA_proxy_validation/MOOC_B_STUDENT_CONDITIONED_HCR_V2
PYTHONPATH=src .venv/bin/python scripts/run_mooc_b_sc_hcr_stage_b.py \
  --output-dir outputs/taskA_proxy_validation/MOOC_B_STUDENT_CONDITIONED_HCR_V2
PYTHONPATH=src .venv/bin/python scripts/freeze_mooc_b_sc_hcr_v2_decision.py --force
```

Synteza claimów + kontrola populacyjna A→G:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mooc_edge_emergence_proxy.py \
  --output-dir outputs/taskA_proxy_validation/mooc_edge_emergence_v1_control
PYTHONPATH=src .venv/bin/python scripts/freeze_proxy_validation_claims.py --force
```


## Analizy pomocnicze

- `analyze_scenario_distributions.py` — rozkłady sześciu scenariuszy i czterech
  szpitali.
- `audit_wave9_input_ranges.py` i `diagnose_wave10_kan_failure.py` — diagnostyka
  wejść oraz residual KAN.
- skrypty `eval_*`, `summarize_*`, `export_*` i `backfill_*` — ewaluacja,
  agregacja oraz eksport już wykonanych runów.

## Historia fal

Nazwy `wave3`–`wave10`, `hcr`, `wnerw` i `panel_b` oznaczają kolejne etapy
rozwoju **Task A**, a nie osobne zadania. Są zachowane dla reprodukowalności
wyników opisanych w raporcie:

`outputs/Wave 0-11 podsumowanie/00_README.md`

Skrypty `run_TaskA_*.sh` oraz `run_wave*.py` w katalogu głównym są historycznymi
entrypointami wcześniejszych fal. Finalny eksperyment Wave 11 korzysta ze
skryptów wymienionych w pierwszej sekcji.

## Zasady uruchamiania

Polecenia wykonuj z katalogu głównego repozytorium i ustaw:

```bash
export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
export PYTHONPATH="$PWD/src:$PWD"
```

Ciężkie wyniki, checkpointy i lokalne logi trafiają do `outputs/` i nie są
wersjonowane. Wyjątkiem jest lekki raport promotorski wskazany wyżej.
