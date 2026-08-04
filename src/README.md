# Mapa kodu Task A

## Główny przepływ

```text
configs/config.yaml
        │
        ▼
train_taskA.py
        ├── data/PreprocessingTaskA/   budowa grafu i kandydatów
        ├── hcr/                       cechy zależności HCR/GHCR
        ├── models/TaskA/              encoder HGT i decodery par
        └── evaluation/                metryki globalne i endpoint-path
```

## Katalogi

- `data/PreprocessingTaskA/` — wczytanie GSN v3, cechy węzłów, negatywne pary,
  splity train/validation/test i graf message passingu.
- `models/TaskA/encoders/` — HeteroSAGE, GATv2 i finalny HGT.
- `models/TaskA/pair_encoders/` — MLP oraz warianty KAN K0–K5.
- `models/TaskA/wave7c_decoder.py` — trzy nieudostępniane encodery ról
  `AZ`, `AG`, `ZG` i finalna fuzja z gałęzią grafową.
- `hcr/` — kolejne wersje HCR/GHCR; `hcr/wave7/wave7c/` odpowiada finalnej
  reprezentacji 40D.
- `evaluation/` — AUPRC, AUROC, Brier, threshold metrics, oversmoothing oraz
  grupowanie według ścieżek do endpointów.
- `experiments/` — wspólne interwencje i funkcje audytów.
- `link_prediction/` i `wnerw/` — historyczne eksperymenty Task A związane ze
  ścieżkami i energią; zachowane dla reprodukowalności fal 5–7.
- `utils/task_a_layout.py` — wspólny porządek indeksów kandydackich węzłów.

## Co czytać najpierw

1. `train_taskA.py`
2. `data/PreprocessingTaskA/load_hetero_recon_data.py`
3. `models/TaskA/hetero_gnn.py`
4. `models/TaskA/encoders/hgt.py`
5. `models/TaskA/wave7c_decoder.py`
6. `models/TaskA/pair_encoders/factory.py`
7. `hcr/wave7/wave7c/attach.py`

Dokładny opis finalnej architektury znajduje się w
`outputs/Wave 0-11 podsumowanie/06_FINAL_MLP_I_KAN_ARCHITEKTURA.md`.
