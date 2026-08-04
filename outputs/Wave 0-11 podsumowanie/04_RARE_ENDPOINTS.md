# Wave 11 — audyt ścieżek do rzadkich endpointów

## Pytanie

Czy globalny zwycięzca MLP i najlepsza architektura KAN zachowują się inaczej
na kandydackich krawędziach należących do ścieżek prowadzących do rzadkich
endpointów klinicznych?

To nadal Task A link prediction. Model nie przewiduje endpointu pacjenta.
Endpoint służy wyłącznie do utworzenia podzbioru kandydackich krawędzi.

## Porównywane modele

- MLP: `TA_MLP`, finalny unshared pair encoder 40→16→8;
- KAN: `TA_KAN_SHALLOW`, najlepsza architektura KAN, 40→8.

Oba używają tego samego HGT, grafu, HCR 40D, kandydatów i splitów.

## Registry endpoint-path

Dla kandydata A→G sprawdzano, czy target G ma skierowaną ścieżkę do endpointu:

`G⇝e`.

Kandydacka krawędź A→G nie była używana do budowania tej osiągalności.
Jedna krawędź mogła należeć do wielu endpointów.

Audyt obejmuje:

- AKI;
- DILI;
- Depression;
- Falls;
- Delirium;
- GI_bleeding;
- Hyponatremia;
- Hyperkalemia;
- QT_arrhythmia;
- Hospitalization.

## Definicja rzadkości

Rzadkość jest oparta na prevalence endpointu wśród pacjentów train konkretnego
scenariusza, nie na prevalence pozytywnych krawędzi.

- `ultra_sparse_<1pct`: mniej niż 1%;
- `sparse_1_5pct`: od 1% do mniej niż 5%;
- `moderate_5_10pct`: od 5% do mniej niż 10%;
- `common_>=10pct`: co najmniej 10%.

W danych nie wystąpiła grupa ultra-sparse. Edge prevalence jest raportowana
osobno, ponieważ odpowiada innemu pytaniu.

## Wyniki macro validation

### Common, prevalence endpointu ≥10%

- 12 komórek scenario×endpoint;
- mean patient prevalence: 18.16%;
- mean positive edges: 28.33;
- MLP AUPRC: 0.97109;
- KAN shallow AUPRC: 0.97260;
- ΔAUPRC KAN−MLP: +0.00151;
- Δnormalized AUPRC: +0.00254.

### Moderate, prevalence 5–10%

- 18 komórek scenario×endpoint;
- mean patient prevalence: 8.06%;
- mean positive edges: 8.0;
- MLP AUPRC: 0.98019;
- KAN shallow AUPRC: 0.98146;
- ΔAUPRC: +0.00329;
- Δnormalized AUPRC: +0.00432.

### Sparse, prevalence 1–5%

- 30 komórek scenario×endpoint;
- mean patient prevalence: 2.84%;
- mean positive edges: 6.4;
- MLP AUPRC: 0.97933;
- KAN shallow AUPRC: 0.98459;
- ΔAUPRC: +0.00526;
- Δnormalized AUPRC: +0.00618.

## Interpretacja

Globalny wynik wszystkich kandydatów nadal wskazuje MLP. KAN shallow ma jednak
lokalną przewagę w podzbiorach endpoint-path, a wielkość przewagi rośnie wraz
ze spadkiem patient endpoint prevalence:

`+0.0015 → +0.0033 → +0.0053 AUPRC`.

Jest to wynik eksploracyjny i podgrupowy:

- liczba pozytywnych krawędzi w części komórek jest mała;
- komórki endpointów nakładają się, bo jedna krawędź może prowadzić do wielu
  endpointów;
- nie wykonano niezależnego testu formalnej istotności;
- selekcja modelu nadal odbywała się na global validation AUPRC;
- wynik nie odwraca globalnej decyzji MLP.

Najbezpieczniejszy wniosek naukowy:

> MLP pozostaje globalnym zwycięzcą Task A. Shallow KAN nie poprawia globalnej
> rekonstrukcji krawędzi, ale wykazuje małą, rosnącą przewagę w audycie ścieżek
> prowadzących do rzadszych endpointów. Efekt wymaga ostrożnej interpretacji ze
> względu na małe i nakładające się podzbiory.

## Pliki źródłowe

- `outputs/wave11_taskA/kan_architecture_audit/endpoint_path/MLP_VS_KAN_RARE_ENDPOINT_TABLE.csv`
- `outputs/wave11_taskA/kan_architecture_audit/endpoint_path/MLP_VS_KAN_RARITY_SUMMARY.csv`
- `outputs/wave11_taskA/kan_architecture_audit/endpoint_path/winner_endpoint_path_metrics_raw.csv`
- `outputs/wave11_taskA/kan_architecture_audit/endpoint_path/RARE_ENDPOINT_AUDIT_MANIFEST.json`
