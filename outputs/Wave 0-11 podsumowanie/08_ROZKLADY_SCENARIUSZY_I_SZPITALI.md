# Porównanie rozkładów scenariuszy i szpitali

## 1. Cel

Analiza pokazuje, że sześć scenariuszy nie jest sześcioma kopiami tego samego
zbioru. Różnice dotyczą selekcji pacjentów, ekspozycji lekowych, obciążeń,
intensywności monitorowania, zaszumienia dokumentacji oraz endpointów.

Osobno analizowana jest heterogeniczność czterech szpitali w scenariuszu
`multihospital`.

## 2. Zakres i metodologia

Główna analiza używa wyłącznie pacjentów train, ponieważ ten podzbiór służy do:

- tworzenia empirical node features;
- dopasowania HCR/GHCR.

Liczebności train:

- clean: 14 000;
- hidden confounder: 14 000;
- selection bias: 9 854;
- no overlap: 14 000;
- noisy documentation: 14 000;
- multihospital: 14 000.

Multihospital:

- hospital 1: 3 564;
- hospital 2: 3 496;
- hospital 3: 3 402;
- hospital 4: 3 538.

Analizowano 154 obserwowalne zmienne z zamrożonego registry typów.
`unobserved_severity` wykluczono, ponieważ jest latentne.

W `noisy_documentation` używano model-facing kolumn `recorded_<variable>`,
jeśli istniały. Dzięki temu analiza opisuje dane faktycznie widziane przez
Task A, a nie ukryte wartości `true`.

## 3. Miary różnicy

### Zmienne binarne

Różnica prevalence:

`Δp = p_scenario − p_clean`.

Raportowane są również Jensen–Shannon divergence dla rozkładów Bernoulliego.

### Zmienne count i continuous

Standardized mean difference:

`SMD = (mean_scenario−mean_clean) / sqrt[(SD_scenario²+SD_clean²)/2]`.

Dodatkowo:

- KS statistic — maksymalna różnica dystrybuant;
- Wasserstein distance — odległość potrzebna do przesunięcia jednej dystrybucji
  w drugą;
- Wasserstein podzielony przez SD clean.

Nie raportowano p-value. Przy kilku–kilkunastu tysiącach pacjentów nawet małe,
nieistotne praktycznie różnice byłyby często statystycznie istotne. Tutaj celem
jest wielkość efektu.

## 4. Globalna charakterystyka scenariuszy

### Hidden confounder

Wszystkie 154 obserwowalne marginale są identyczne z `clean`.

To nie jest błąd. Scenariusz zmienia rolę ukrytego `unobserved_severity`, ale
nie zmienia obserwowanych kolumn wejściowych. Różnica ma charakter latentny i
przyczynowy, a nie marginalny.

Wniosek:

> Nie każdy stress scenario musi być widoczny w histogramach obserwowanych
> zmiennych. Hidden confounder jest testem niewidocznej zmiany mechanizmu.

### Selection bias

To najbardziej wyraźny targeted shift.

Największe różnice względem clean:

- `creatinine_tested`: 48.69% → 69.17%, czyli +20.48 pp;
- `lft_tested`: 32.93% → 46.78%, +13.85 pp;
- `hospital_contact`: 27.09% → 38.48%, +11.40 pp;
- `monitoring_intensity`: SMD +0.246;
- `Hospitalization`: 22.50% → 24.88%, +2.38 pp;
- `AKI`: 13.10% → 14.61%, +1.51 pp.

Interpretacja: selekcja wzbogaca kohortę w pacjentów częściej monitorowanych,
testowanych i hospitalizowanych. Zmiana nie jest jedynie spadkiem liczebności
z 14 000 do 9 854; zmienia skład kliniczny train.

### No overlap

Zmiana jest skupiona w ekspozycjach i obciążeniu krwawieniowym:

- `antiplatelet`: 29.71% → 35.25%, +5.54 pp;
- `ddi_bleeding_dual`: 8.41% → 13.21%, +4.80 pp;
- `anticoagulant`: 27.25% → 31.97%, +4.72 pp;
- `bleeding_risk_load`: SMD +0.102;
- `active_drug_count`: +0.167 leku, SMD +0.075.

Endpointy jako grupa zmieniają się mniej niż upstream exposure/load variables.
Scenariusz przede wszystkim zmienia support i overlap kombinacji leczenia.

### Noisy documentation

Zmienia model-facing recorded values, mimo braku dodatkowych NaN.

Największe różnice:

- `dizziness`: 14.31% → 9.38%, −4.93 pp;
- `mood_lowering`: 8.98% → 5.84%, −3.14 pp;
- `Depression`: 8.46% → 5.60%, −2.86 pp;
- `Hospitalization`: 22.50% → 19.94%, −2.56 pp;
- `Falls`: 7.79% → 6.31%, −1.48 pp.

Dla bardzo rzadkich endpointów część recorded rates wzrasta:

- Serotonin syndrome: 0.25% → 1.26%;
- Rhabdomyolysis: 0.34% → 1.30%;
- Lactic acidosis: 0.55% → 1.49%.

Interpretacja: documentation noise nie jest klasycznym missingness shift.
Powoduje false negatives i false positives w recorded variables.

### Multihospital — pooled vs clean

Po połączeniu czterech szpitali różnice wyglądają umiarkowanie:

- `active_drug_count`: +0.176, SMD +0.078;
- `qt_drug_load_v3`: SMD +0.059;
- `ddi_qt_multidrug_load`: SMD +0.057;
- `bleeding_risk_load`: SMD +0.052;
- `antiplatelet`: +1.89 pp;
- `acei_arb`: +1.53 pp;
- `macrolide`: +1.53 pp;
- `ssri`: +1.51 pp.

Pooled distribution maskuje jednak silniejsze różnice wewnątrz szpitali.

## 5. Heterogeniczność wewnątrz multihospital

Każdy szpital porównano z połączonymi trzema pozostałymi szpitalami.

### Hospital 1

Najbardziej zbliżony do pooled population.

- active drug count: −0.189 vs inne, SMD −0.084;
- QT multidrug load: SMD −0.087;
- hospital contact: −2.17 pp;
- większość efektów mała.

### Hospital 2

Profil wysokiej ekspozycji i intensywności leczenia.

- active drug count: +0.871 vs inne, SMD +0.383;
- bleeding risk load: SMD +0.203;
- QT multidrug load: SMD +0.189;
- QT drug load: SMD +0.186;
- nephrotoxin load: SMD +0.179;
- diuretic: +5.21 pp;
- creatinine tested: +5.18 pp;
- electrolytes tested: +5.02 pp.

Endpointy:

- AKI 14.56% — najwyższe spośród szpitali;
- Hospitalization 23.66% — najwyższe;
- Hyponatremia 6.95% — najniższe.

### Hospital 3

Profil niskiej ekspozycji i słabszego monitorowania.

- active drug count: −1.026 vs inne, SMD −0.464;
- bleeding risk load: SMD −0.250;
- hepatic drug load: SMD −0.207;
- QT drug load: SMD −0.198;
- serotonergic load: SMD −0.190;
- electrolytes tested: −7.41 pp;
- creatinine tested: −7.26 pp;
- ECG performed: −6.82 pp;
- LFT tested: −6.54 pp.

Endpointy:

- AKI 12.38% — najniższe;
- Depression 8.91% — najwyższe;
- Hyponatremia 8.55% — najwyższe;
- QT arrhythmia 1.29% — najniższe.

### Hospital 4

Umiarkowanie wyższa ekspozycja niż pooled others.

- active drug count: +0.324, SMD +0.142;
- serotonergic load: SMD +0.086;
- QT drug load: SMD +0.083;
- creatinine tested: +3.68 pp;
- metformin: +3.49 pp;
- LFT tested: +3.30 pp.

Endpointy:

- Hyperkalemia 5.43% — najwyższe;
- Rhabdomyolysis 0.45% — najwyższe;
- Lactic acidosis 0.28% — najniższe.

## 6. Podsumowanie wielkości shiftu

P90 absolutnego SMD dla zmiennych count/continuous:

- hidden confounder: 0.000;
- selection bias: 0.035;
- no overlap: 0.047;
- noisy documentation: 0.000, ponieważ targeted changes są głównie binarne;
- multihospital pooled: 0.058.

P90 absolutnej różnicy prevalence binarnych:

- hidden confounder: 0.00 pp;
- selection bias: 0.90 pp, ale maksimum 20.48 pp;
- no overlap: 0.62 pp, maksimum 5.54 pp;
- noisy documentation: 0.62 pp, maksimum 4.93 pp;
- multihospital pooled: 0.72 pp, maksimum 1.89 pp.

Wewnątrz multihospital P90 SMD:

- hospital 1: 0.083;
- hospital 2: 0.194;
- hospital 3: 0.224;
- hospital 4: 0.084.

Hospital 2 i 3 tworzą największy kontrast.

## 7. Najważniejszy wniosek do prezentacji

> Scenariusze zachowują ten sam graf prawdy, ale nie są dystrybucyjnie
> równoważne. Selection bias zmienia intensywność monitorowania i skład kohorty,
> no overlap przesuwa ekspozycje i burden, noisy documentation zmienia observed
> labels bez wzrostu missingness, a multihospital zawiera przeciwstawne profile
> ekspozycji szpitali 2 i 3. Hidden confounder pozostaje celowo niewidoczny w
> obserwowanych marginalach, ponieważ jest zmianą latentnego mechanizmu.

## 8. Pliki wynikowe

Katalog:

`outputs/Wave 0-11 podsumowanie/scenario_distribution_audit/`

Najważniejsze:

- `scenario_shift_summary.csv`;
- `scenario_top20_shifts.csv`;
- `scenario_shifts_vs_clean.csv`;
- `scenario_variable_distributions.csv`;
- `endpoint_prevalence_scenarios_and_hospitals.csv`;
- `multihospital_shift_summary.csv`;
- `multihospital_top20_shifts.csv`;
- `multihospital_shifts_vs_other_hospitals.csv`;
- `multihospital_variable_distributions.csv`;
- `MANIFEST.json`.

Kod:

`scripts/analyze_scenario_distributions.py`.
