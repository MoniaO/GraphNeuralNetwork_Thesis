# Task A — historia eksperymentów Wave 0–11

Pakiet przygotowany jako materiał na spotkanie z promotorem. Zakres obejmuje
wyłącznie Task A: rekonstrukcję krawędzi audytowanego grafu przyczynowego.

## Pliki

- `01_DANE_WEJSCIOWE_I_PIPELINE.md` — zbiór, splity, wektory, topologia,
  encodery, dekodery i batching.
- `02_HISTORIA_FAL_0_11.md` — chronologiczna historia pytań, runów, ablacji,
  wyników i decyzji.
- `03_WZORY_I_STATYSTYKA.md` — wszystkie główne wzory z objaśnieniem symboli.
- `04_RARE_ENDPOINTS.md` — końcowe porównanie MLP–KAN dla ścieżek prowadzących
  do rzadkich endpointów.
- `06_FINAL_MLP_I_KAN_ARCHITEKTURA.md` — dokładny opis i diagramy finalnego MLP
  oraz najlepszego KAN shallow.
- `07_MAPA_KODU_I_KOLEJNOSC_CZYTANIA.md` — struktura repozytorium, przepływ
  wywołań i kolejność plików do analizy.
- `08_ROZKLADY_SCENARIUSZY_I_SZPITALI.md` — train-only distribution audit dla
  sześciu scenariuszy i czterech szpitali multihospital.
- `scenario_distribution_audit/` — komplet szczegółowych CSV z rozkładami,
  shiftami, endpointami i rankingami zmiennych.
- `wave_0_11_experiments.csv` — główna tabela eksperymentów, gotowa do Excela.
- `wave_0_11_key_results.csv` — najważniejsze wyniki liczbowe w długim formacie.

## Ważne uwagi interpretacyjne

1. Wave 0 była fazą koncepcyjną i walidacją pipeline; nie zachował się osobny
   zestaw runów Wave 0. Formalne wyniki zaczynają się od Wave 1.
2. W repozytorium nie istnieje Wave 8. Numeracja przechodzi z Wave 7D do Wave 9.
3. `training.batch_size: 32` z ogólnego YAML nie jest używane przez
   `train_taskA.py`. Główny trening GNN jest full-batch: jeden krok optymalizatora
   na cały graf i wszystkie kandydaty train w każdej epoce.
4. Test był report-only. Checkpointy wybierano na podstawie validation AUPRC.
5. Nie wykonywano formalnych testów p-value w głównym pipeline. Wnioski oparto
   na mean±SD, paired deltas, liczbie wygranych seedów, progach decyzyjnych oraz
   miejscami bootstrap CI.

## Główne źródła

- `README.md`
- `outputs/taskA_wave_program_table.csv`
- `outputs/wave1_baselines_ablations/`
- `outputs/wave2_architecture_2026-07-31*/`
- `outputs/wave3_hcr_2026-07-31/`
- `outputs/wave3b_motif_2026-07-31/`
- `outputs/wave4c_latent_2026-07-31/`
- `outputs/wave4d_context_role_audit_2026-07-31/`
- `outputs/wave5*`, `outputs/wave6_hcr/`, `outputs/wave7/`
- `outputs/wave9/`, `outputs/wave10/`, `outputs/wave11_taskA/`
- `configs/`, `src/train_taskA.py`, `src/models/TaskA/`, `src/hcr/`

Gdy artefakt liczbowy nie istnieje lub jest niekompletny, raport oznacza to
wprost jako „brak”, „planowane” albo „nieudokumentowane”.
