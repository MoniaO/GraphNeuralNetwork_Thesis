# Finalne architektury Task A: MLP i najlepszy KAN

## 1. Co jest wspólne

Oba modele rozwiązują dokładnie ten sam problem:

`P(A→G należy do G_true | G_train, cechy węzłów, HCR/GHCR)`.

W obu modelach identyczne są:

- dane GSN v3;
- patient split i candidate edge split;
- `G_train` używany do message passingu;
- empiryczne cechy węzłów liczone tylko na pacjentach train;
- HCR/GHCR dopasowane tylko na pacjentach train danego scenariusza;
- HGT L3, 8 heads, hidden 32;
- residual connections, leaky ReLU i HGT dropout 0.2;
- pełny motyw `AZ⊕AG⊕ZG`;
- trzy niewspółdzielone encodery ról;
- graph branch i final output head;
- loss, optimizer, early stopping i selection po validation AUPRC.

Jedyna zmiana architektoniczna znajduje się wewnątrz trzech encoderów par HCR:

- finalny MLP: `Linear(40→16→8)`;
- najlepszy KAN K1: `KANLinear(40→8)`.

## 2. Wejścia

### 2.1 Wejście grafowe

Dla każdego węzła `v`:

`x_v = [mean_train(v), std_train(v), missing_rate_train(v)]`.

HGT przetwarza heterogeniczny `G_train` i zwraca:

`z_v∈R^32`.

Dla kandydackiej krawędzi A→G:

`q_AG = z_A ⊕ z_G ⊕ (z_A⊙z_G) ⊕ |z_A−z_G| ∈ R^128`.

### 2.2 Wejście HCR/GHCR

Dla kandydata A→G wybierany jest kontekst Z. Powstają trzy role:

- AZ — source–context;
- AG — direct source–target;
- ZG — context–target.

Każda rola ma 40D:

- 16 współczynników macierzy Legendre 4×4;
- 4 summary energii zależności;
- 4 marginale pierwszej zmiennej;
- 4 marginale drugiej zmiennej;
- 4 joint-activity/rarity/lift-like;
- 1 complete-case support ratio;
- 1 flaga support;
- 3 one-hot typu pierwszej zmiennej;
- 3 one-hot typu drugiej zmiennej.

Pełne surowe wejście HCR:

`h_motif = h_AZ ⊕ h_AG ⊕ h_ZG ∈ R^120`.

## 3. Wspólna gałąź grafowa

Kod: `src/models/TaskA/wave7c_decoder.py`.

`q_AG(128)`

`→ Linear(128,128)`

`→ GELU`

`→ Dropout(0.2)`

`→ Linear(128,64)`

`→ GELU`

`→ LayerNorm(64)`

`→ g_graph∈R^64`.

## 4. Finalny MLP

Model: `TA_MLP`.

Config: `configs/taskA/mlp_full_retrain.yaml`.

### Encoder pojedynczej roli

Każda rola ma oddzielne parametry:

`E_r^MLP(h_r) = LayerNorm_8(Linear_16→8(Dropout_0.1(LayerNorm_16(GELU(Linear_40→16(h_r))))))`.

Czytelny zapis:

`40D`

`→ Linear(40,16)`

`→ GELU`

`→ LayerNorm(16)`

`→ Dropout(0.1)`

`→ Linear(16,8)`

`→ LayerNorm(8)`.

Trzy niezależne egzemplarze:

- `E_AZ^MLP`;
- `E_AG^MLP`;
- `E_ZG^MLP`.

Po konkatenacji:

`g_HCR^MLP = E_AZ(h_AZ) ⊕ E_AG(h_AG) ⊕ E_ZG(h_ZG) ∈ R^24`.

### Finalna głowa

`g_final = g_graph ⊕ g_HCR^MLP ∈ R^88`.

`88`

`→ Linear(88,64)`

`→ GELU`

`→ Dropout(0.2)`

`→ Linear(64,1)`

`→ logit l_AG`

`→ sigmoid`

`→ P(A→G)`.

### Diagram finalnego MLP

```mermaid
flowchart LR
    P[Pacjenci train] --> X[Empirical node vectors<br/>mean, std, missing]
    X --> HGT[HGT<br/>L3, H8, hidden 32<br/>leaky ReLU, residual, drop 0.2]
    GT[G_train<br/>positive train edges + reverse] --> HGT
    HGT --> ZA[z_A: 32D]
    HGT --> ZG[z_G: 32D]
    ZA --> Q[q_AG = z_A ⊕ z_G ⊕ z_A⊙z_G ⊕ |z_A-z_G|<br/>128D]
    ZG --> Q
    Q --> GB[Graph branch<br/>128→128→64<br/>GELU, drop 0.2, LN]

    P --> HF[Train-only HCR/GHCR fit]
    HF --> AZ[h_AZ: 40D]
    HF --> AG[h_AG: 40D]
    HF --> CZ[h_ZG: 40D]
    AZ --> EAZ[MLP_AZ<br/>40→16→8<br/>GELU, LN, drop 0.1]
    AG --> EAG[MLP_AG<br/>40→16→8<br/>GELU, LN, drop 0.1]
    CZ --> EZG[MLP_ZG<br/>40→16→8<br/>GELU, LN, drop 0.1]
    EAZ --> M[Concatenate role latents<br/>8+8+8 = 24D]
    EAG --> M
    EZG --> M

    GB --> F[Final latent<br/>64+24 = 88D]
    M --> F
    F --> O[Output head<br/>88→64→1<br/>GELU, drop 0.2]
    O --> PR[Sigmoid probability<br/>P(A→G)]
```

### Rozmiar i wynik

- parametry: 179 643;
- Wave 11 mean validation AUPRC po scenariuszach: około 0.953;
- globalny zwycięzca.

## 5. Najlepszy KAN: K1 shallow

Model: `TA_KAN_SHALLOW`.

Config: `configs/taskA/kan_architectures/k1_shallow.yaml`.

Kod: `src/models/TaskA/pair_encoders/kan_shallow.py`.

### Dlaczego shallow

Bezpośredni dwuwarstwowy KAN `40→16→8` przegrał z MLP. Audyt K1–K5 wykazał,
że najmniejszą stratę i najlepszą stabilność ma jednowarstwowy KAN:

`KANLinear(40,8)`.

Model usuwa pośrednie 16D i pozwala każdemu wyjściowemu latentowi bezpośrednio
uczyć addytywne funkcje wszystkich 40 wejść.

### Encoder pojedynczej roli

`E_r^KAN(h_r) = LayerNorm_8(KANLinear_40→8(h_r))`.

KANLinear:

`o_j = Σ_i [w_ji^base·SiLU(x_i) + w_ji^spline·Σ_m c_jim B_m(x_i)] + b_j`.

Znaczenie:

- `x_i` — i-ta cecha 40D;
- `o_j` — j-ty z ośmiu kanałów latentnych;
- `B_m` — funkcja bazowa B-spline;
- `c_jim` — współczynniki spline;
- część bazowa używa SiLU;
- część spline uczy lokalne nieliniowe kształty.

Ustawienia:

- `grid_size=5`;
- `spline_order=3`;
- `grid_range=[−3,3]`;
- `grid_update=false`;
- `base_scale_init=1.0`;
- `spline_scale_init=0.1`;
- `spline_l1=10^−5`;
- bias włączony;
- LayerNorm na wyjściu 8D.

Ważne: config zawiera `dropout: 0.2`, ale `KANShallowPairEncoder` nie wywołuje
Dropoutu. Wewnątrz K1 jest tylko `KANLinear(40→8)→LayerNorm(8)`. Dropout 0.2
pozostaje aktywny w HGT, graph branch i output head.

Trzy niezależne egzemplarze:

- `E_AZ^KAN`;
- `E_AG^KAN`;
- `E_ZG^KAN`.

`g_HCR^KAN = E_AZ(h_AZ) ⊕ E_AG(h_AG) ⊕ E_ZG(h_ZG) ∈ R^24`.

Finalna głowa jest identyczna z MLP:

`[g_graph(64),g_HCR^KAN(24)] → 88→64→1`.

### Diagram najlepszego KAN

```mermaid
flowchart LR
    P[Pacjenci train] --> X[Empirical node vectors<br/>mean, std, missing]
    X --> HGT[HGT<br/>L3, H8, hidden 32<br/>leaky ReLU, residual, drop 0.2]
    GT[G_train<br/>positive train edges + reverse] --> HGT
    HGT --> ZA[z_A: 32D]
    HGT --> ZT[z_G: 32D]
    ZA --> Q[q_AG: 128D<br/>concat, product, absolute difference]
    ZT --> Q
    Q --> GB[Graph branch<br/>128→128→64<br/>GELU, drop 0.2, LN]

    P --> HF[Train-only enriched HCR/GHCR]
    HF --> AZ[h_AZ: 40D]
    HF --> AG[h_AG: 40D]
    HF --> CZ[h_ZG: 40D]
    AZ --> EAZ[KAN_AZ shallow<br/>KANLinear 40→8<br/>cubic spline, grid 5, LN]
    AG --> EAG[KAN_AG shallow<br/>KANLinear 40→8<br/>cubic spline, grid 5, LN]
    CZ --> EZG[KAN_ZG shallow<br/>KANLinear 40→8<br/>cubic spline, grid 5, LN]
    EAZ --> M[Concatenate role latents<br/>8+8+8 = 24D]
    EAG --> M
    EZG --> M

    GB --> F[Final latent<br/>64+24 = 88D]
    M --> F
    F --> O[Ten sam output head<br/>88→64→1<br/>GELU, drop 0.2]
    O --> PR[Sigmoid probability<br/>P(A→G)]
```

### Rozmiar i wynik

- parametry: 188 235;
- o 8 592 parametrów więcej niż finalny MLP;
- mean paired Δvalidation vs MLP po 18 runach: −0.004436±0.008917;
- wygrał 5/18 pojedynczych paired runów;
- najlepszy scenariusz: `no_overlap`, Δ +0.000803;
- najsłabszy: `selection_bias`, Δ −0.014591.

## 6. Porównanie warstwa po warstwie

Wspólne:

1. pacjenci train → empirical node vectors;
2. `G_train` + vectors → HGT → 32D node embeddings;
3. A,G embeddings → 128D graph pair vector;
4. graph branch 128→128→64;
5. train patients → enriched HCR/GHCR;
6. role blocks AZ, AG, ZG po 40D;
7. trzy niewspółdzielone role → trzy latenty 8D;
8. 64D graph + 24D HCR → 88D;
9. output head 88→64→1.

Różnica:

- MLP: każda rola `Linear 40→16→8`, GELU, LN i dropout 0.1;
- KAN K1: każda rola `KANLinear 40→8` + LN, spline L1, bez wewnętrznego
  Dropoutu.

## 7. Wniosek

MLP jest lepszym wyborem finalnym, ponieważ:

- ma wyższy globalny validation AUPRC;
- ma mniej parametrów;
- jest stabilniejszy szczególnie w `selection_bias`;
- KAN nie przekroczył progu globalnej promocji.

KAN shallow pozostaje wartościowym modelem audytowym, ponieważ:

- był najlepszym z K1–K5;
- niemal zrównał się z MLP w części scenariuszy;
- dawał rosnącą lokalną przewagę na ścieżkach do rzadszych endpointów;
- jego addytywne spline mogą być analizowane jako funkcje cech 40D.

Formalny model finalny: `TA_MLP`.

Najlepszy model KAN do analiz porównawczych: `TA_KAN_SHALLOW`.
