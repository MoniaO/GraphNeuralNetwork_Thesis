# Skrypty eksperymentalne Task A

Wszystkie skrypty w tym katalogu dotyczą rekonstrukcji krawędzi (link
prediction). Kod modelu znajduje się w `src/`, a skrypty jedynie uruchamiają,
audytują lub agregują eksperymenty.

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
