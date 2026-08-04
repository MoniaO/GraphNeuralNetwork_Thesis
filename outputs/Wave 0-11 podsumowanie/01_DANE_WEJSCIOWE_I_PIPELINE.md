# Dane wejściowe i pipeline Task A

## 1. Cel zadania

Task A jest heterogenicznym link prediction. Dla pary węzłów `(s,t)` model
zwraca prawdopodobieństwo, że skierowana krawędź `s→t` należy do audytowanego
grafu przyczynowego `G_true`.

Pacjent nie jest przykładem treningowym GNN. Wiersze pacjentów służą do:

1. utworzenia empirycznych cech węzłów;
2. dopasowania cech zależności HCR/GHCR w późniejszych falach.

Przykładem treningowym jest kandydacka para węzłów z etykietą `y∈{0,1}`.

## 2. Zbiór GSN v3

Źródło danych:

`~/Desktop/GSN Graphs dysertation 2026/2 v3. Data/dataset_v3/`

Podstawowa charakterystyka:

- 20 000 pacjentów w scenariuszu `clean`;
- 14 000 pacjentów train, 3 000 validation i 3 000 test;
- 157 kolumn w pliku pacjentów (`patient_id` + około 156 zmiennych);
- 155 zadeklarowanych węzłów, w tym 154 obserwowalne;
- jeden latentny węzeł `unobserved_severity`, wyłączony z message passingu;
- 405 audytowanych krawędzi w pełnym DAG;
- 366 dodatnich krawędzi po filtracji węzłów latentnych;
- jeden oczekiwany izolowany węzeł `cognitive_slowing`.

Scenariusz `selection_bias` zawiera 14 062 pacjentów. Pozostałe scenariusze
reprezentują ten sam graf prawdy, ale inny świat obserwacji.

### Typy obserwowalnych węzłów

- `mechanism`: 49;
- `drug_exposure`: 29;
- `adr_or_intermediate_state`: 28;
- `patient_context`: 24;
- `clinical_endpoint`: 13;
- `observation_or_selection`: 12.

### Warstwy DAG

- `1_patient_context`: 23;
- `2_drugs`: 29;
- `3_mechanisms`: 49;
- `4_intermediate_states`: 28;
- `5_observation_selection`: 12;
- `6_endpoints`: 13.

### Najliczniejsze typy krawędzi wśród 366 obserwowalnych pozytywów

- `risk_modifier`: 47;
- `interaction_amplify`: 43;
- `drug_to_burden`: 32;
- `drug_to_load`: 32;
- `drug_to_mechanism`: 25;
- `treatment_burden`: 24;
- `mechanism_to_adr`: 23;
- `adr_to_endpoint`: 20;
- `interaction_gate_component`: 19;
- `adr_burden_component`: 16.

## 3. Sześć scenariuszy pacjentów

- `clean` — bazowy zapis bez celowego zaburzenia;
- `hidden_confounder` — wpływ nieobserwowanego nasilenia;
- `selection_bias` — selekcja pacjentów;
- `no_overlap` — ograniczony overlap grup;
- `noisy_documentation` — zaszumione kolumny `recorded_*`;
- `multihospital` — heterogeniczność czterech szpitali.

Graf `G_true`, kandydaci i split krawędzi pozostają wspólne. Zmienia się
macierz pacjentów, a więc empiryczne cechy węzłów i HCR dopasowane na train.

## 4. Dwa niezależne splity — ochrona przed leakage

### Split pacjentów

Deterministyczny split 70/15/15:

- train: 14 000;
- validation: 3 000;
- test: 3 000.

W Task A do cech węzłów i HCR używani są wyłącznie pacjenci train.
Patient split był balansowany wieloetykietowo po 13 endpointach.

### Split kandydackich krawędzi

Pozytywy to 366 obserwowalnych krawędzi. Negatywy są losowane jako
nieistniejące pary zgodne z dozwolonymi parami typów, w proporcji 3:1.

- pozytywy: 366;
- negatywy: 1 098;
- wszystkie kandydaty: 1 464;
- train: około 1 025, w tym około 256 pozytywów;
- validation: około 220;
- test: około 220.

Split jest stratyfikowany 70/15/15 po etykiecie. `candidate_seed=20260722`
zamraża negatywy i splity między scenariuszami.

Do message passingu trafiają wyłącznie dodatnie krawędzie train i ich relacje
odwrotne, gdy `add_reverse_edges=true`. Pozytywne krawędzie validation/test są
ukryte. Negatywy nigdy nie tworzą topologii message passingu.

## 5. Surowy wektor pacjenta

Wiersz macierzy pacjentów można zapisać jako:

`p_i = [v_i1, v_i2, ..., v_iD]`,

gdzie `i` oznacza pacjenta, `D≈156` zmiennych klinicznych, a `v_ij` jest
wartością zmiennej `j`. Rejestr `src/hcr/variable_specs_v3.py` przypisuje
zmienne do typów:

- binary — ekspozycje, konteksty, stany i endpointy 0/1;
- count — liczby obciążeń/zdarzeń;
- continuous — m.in. ciągłe parametry kliniczne;
- dodatkowe zmienne mechanizmów, obserwacji, obciążeń i bramek interakcji.

Nie wszystkie zmienne pacjenta są podawane bezpośrednio do GNN.

## 6. Empiryczny wektor węzła

Dla obserwowalnego węzła `v` główny profil `empirical` tworzy:

`x_v = [mean_train(v), std_train(v), missing_train(v)]`.

Znaczenie:

- `mean_train(v)` — średnia wartości wśród pacjentów train; dla zmiennej
  binarnej jest to prevalence;
- `std_train(v)` — odchylenie standardowe populacyjne (`ddof=0`);
- `missing_train(v)` — udział braków danych.

W `noisy_documentation` używana jest kolumna `recorded_<name>`, jeśli istnieje.
Ciągłe kanały są standaryzowane z-score w macierzy cech węzłów.

Profile kontrolne:

- `structural` / `topology_only` — stała cecha, bez sygnału pacjentów;
- `empirical_shuffled` — empiryczne wektory przypisane losowo;
- `empirical_layer` — empirical + one-hot warstwy;
- `empirical_ontology` — empirical + warstwa/severity/is_endpoint;
- `oracle` — metadane generatora; nie jest głównym baseline.

## 7. Ewolucja HCR/GHCR

### Wave 3: binary compact, 8D

Dla pary binarnej wejście zawierało osiem statystyk:

`[p11, P(V=1|U=1), risk_difference, log_odds_ratio, phi, MI, support, uncertainty]`.

### Wave 4–5D: motyw 24D

Trzy role par:

`AZ ⊕ AG ⊕ ZG`,

każda po 8D, razem 24D. W latent pairwise używano relacji pomiędzy obserwowanymi
zmiennymi A, B i Y, bez odczytu ukrytej bramki G.

### Wave 7 Panel B i Wave 9–11: enriched 40D na rolę

Każdy blok pary ma 40 wymiarów:

- sloty 0–15: spłaszczona macierz współczynników Legendre 4×4;
- sloty 16–19: energia całkowita, średnia energia, udział low-frequency,
  maksymalna wartość bezwzględna;
- sloty 20–23: marginale zmiennej U;
- sloty 24–27: marginale zmiennej V;
- sloty 28–31: joint activity, rarity i statystyki lift-like;
- slot 32: `n_complete/n_train`;
- slot 33: flaga support;
- sloty 34–36: one-hot typu U;
- sloty 37–39: one-hot typu V.

Pełny motyw:

`h_motif = h_AZ ⊕ h_AG ⊕ h_ZG ∈ R^120`.

HCR/GHCR jest za każdym razem dopasowywane wyłącznie na pacjentach train
konkretnego scenariusza.

## 8. Wykorzystanie topologii grafu

Encoder GNN wykonuje message passing po heterogenicznym `G_train`. Typ relacji
określa transformację/agregację. W kolejnych falach testowano:

- HeteroSAGE — `SAGEConv` per relacja;
- Explicit R-GCN — wspólna warstwa z identyfikatorem relacji;
- GATv2 — attention na relacjach;
- HGT — heterogeniczny attention zależny od typu węzła i relacji.

Wszystkie modele link prediction tworzą embeddingi węzłów `z_s` i `z_t`.
Podstawowy wektor pary grafowej to:

`q_st = [z_s, z_t, z_s⊙z_t, |z_s-z_t|]`.

Od Wave 3 decoder dokleja do gałęzi grafowej HCR. Od Wave 7C trzy role HCR mają
osobne encodery, a ich latenty są łączone z latentem grafowym.

## 9. Encodery i dekodery

### Wave 1

Encodery: HeteroSAGE i R-GCN. Decoder:

`q_st → MLP → logit`.

### Wave 2–3

Zamrożony encoder HGT L1, 8 heads, hidden 64. Od Wave 3:

`[q_st, h_HCR] → MLP → logit`.

### L2 audit i Wave 7

HGT L3, 8 heads, hidden 32, leaky ReLU, residual, dropout 0.2.
Decoder [256,128] dla 24D, potem Wave7C decoder z gałęzią grafową i osobnymi
encoderami motywu.

### Finalny stack Wave 7D / 9–11

- HGT: L3, heads 8, hidden 32, leaky ReLU, residual, dropout 0.2;
- grafowy wektor pary: 128D;
- grafowa gałąź: projekcja do 64D;
- trzy niewspółdzielone role AZ, AG, ZG;
- każda rola 40D→16D→8D w MLP lub odpowiedniku KAN;
- latent HCR: 24D;
- finalny latent: 64D⊕24D = 88D;
- końcowy MLP klasyfikacyjny zwraca jeden logit.

MLP pary:

`Linear(40,16) → GELU → LayerNorm → Dropout → Linear(16,8) → LayerNorm`.

KAN direct:

`KANLinear(40,16) → LayerNorm → Dropout → KANLinear(16,8) → LayerNorm`.

KAN shallow:

`KANLinear(40,8) → LayerNorm`.

## 10. Trening i liczba batchy

Główna pętla `src/train_taskA.py` jest full-batch:

1. cały heterograf trafia do encodera;
2. wszystkie kandydaty train są oceniane jednocześnie;
3. liczony jest jeden loss;
4. wykonywany jest jeden `optimizer.step()` na epokę.

Zatem:

- batchy na epokę: 1;
- kroków optymalizatora: liczba faktycznie wykonanych epok;
- `training.batch_size: 32` nie wpływa na ten pipeline.

Typowo:

- Wave 1: limit 100 epok, patience 20;
- Wave 5D: osobny MLP krawędzi, minibatch 256 krawędzi, 25 epok;
- L2 audit / Wave 7: do 300 epok, patience 40;
- Wave 9–11: do 200 epok, patience 40.

Wave 5A–5C oraz Wave 6 wykonywały obliczenia ścieżek/cech i nie trenowały GNN.

## 11. Fairness i fingerprinty

Pipeline zapisuje:

- `candidate_fingerprint` — pary, etykiety i splity;
- `feature_fingerprint` — tensory cech węzłów;
- `graph_fingerprint` — krawędzie message passingu;
- `patient_train_fingerprint` — macierz pacjentów użyta do HCR;
- `candidate_registry_sha256` — zamrożone registry Wave 5D.

W Wave 11 MLP i KAN miały zero mismatchy fingerprintów. Dla danego
`scenario+seed` różniła się architektura pair encodera, a nie dane.
