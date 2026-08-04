# Konfiguracje Task A

Projekt używa Hydra. Główny plik `config.yaml` składa konfigurację danych,
modelu i wariantu HCR.

## Najważniejsze grupy

- `data/dataset_v3.yaml` — główny zbiór GSN v3 i scenariusz pacjentów.
- `model/TaskA_hgt_final.yaml` — publiczny alias finalnego HGT
  (domyślnie `decoder.arch=unshared_mlp`).
- `taskA/mlp_full_retrain.yaml` — finalna kontrola MLP.
- `taskA/kan_full_retrain.yaml` — bezpośredni odpowiednik KAN.
- `taskA/kan_architectures/` — warianty audytu K0–K9; pełny screening dotyczy
  K1–K5.
- `hcr/final_ghcr.yaml` — publiczny alias finalnych cech GHCR 3 × 40D.
- `hcr/`, `wnerw/` i `link_prediction/` — konfiguracje wcześniejszych fal
  zachowane dla reprodukowalności.

Aliasy wskazują odpowiednio na historyczne źródła
`model/TaskA_hgt_wave7c.yaml` i `hcr/w7c_b2_audit.yaml`.

## Sprawdzenie kompozycji

```bash
PYTHONPATH=src .venv/bin/python src/train_taskA.py \
  --cfg job \
  model=TaskA_hgt_final \
  hcr=final_ghcr \
  wandb.enabled=false
```

Polecenie wyświetla złożoną konfigurację bez uruchamiania treningu.
