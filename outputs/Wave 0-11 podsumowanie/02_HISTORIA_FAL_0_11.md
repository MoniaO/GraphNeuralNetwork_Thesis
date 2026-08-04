# Historia eksperymentów Wave 0–11

## Wave 0 — przygotowanie problemu

Wave 0 nie występuje jako osobny katalog z metrykami. Była fazą koncepcyjną:

- ustalono Task A jako link prediction na audytowanym DAG;
- usunięto oracle features generatora z głównego wejścia;
- przyjęto empirical node vector z pacjentów train;
- ujednolicono globalny layout indeksów węzłów;
- rozdzielono patient split od edge candidate split;
- zamrożono `candidate_seed=20260722`.

Wniosek: pipeline był gotowy do formalnego baseline Wave 1. Nie należy
przypisywać Wave 0 osobnego AUPRC.

## Wave 1 — baseline i ablacje metodologiczne

### BAZA: głębokość i scenariusze

Uruchomiono 48 runów: 6 scenariuszy × 2 modele × 4 głębokości.

Modele:

- HeteroSAGE, hidden 32, L1–L4;
- R-GCN, L1–L4;
- bez HCR;
- empirical node vectors;
- decoder oparty o `q_st=[z_s,z_t,z_s⊙z_t,|z_s-z_t|]`.

Clean validation AUPRC:

- SAGE L1 0.658, L2 0.692, L3 0.637, L4 0.599;
- R-GCN: najlepszy L1, 0.615.

Zamrożono SAGE L2 i R-GCN L1. Po zamrożeniu głębokości scenariusze miały mały
wpływ: dla SAGE `|Δ|≤0.002`, dla R-GCN `|Δ|≤0.009`.

### Feature ablation

Najpierw wykonano 6 runów diagnostycznych, potem 30 runów multi-seed:
2 modele × 3 profile × 5 seedów.

Porównano empirical, topology-only i empirical-shuffled. W multi-seed:

- SAGE: Δ(emp−topo) +0.031±0.023; Δ(emp−shuffle) +0.020±0.032;
- R-GCN: +0.053±0.058 i +0.158±0.053.

### E1: frozen feature ablation

Model trenowano na empirical, a następnie bez retrainu podmieniano wejście.

- SAGE: spadek względem empirical około 0.22±0.11 AUPRC;
- R-GCN: około 0.26±0.09;
- empirical wygrało 5/5 seedów.

To pokazało, że wytrenowany model realnie używa cech pacjentów.

### E2: permutacja typów węzłów

Największe straty po permutacji:

- mechanism: +0.086 na korzyść niepermutowanego wejścia;
- patient_context: +0.061;
- ADR/intermediate: +0.028.

### E3: wyniki per edge type

Najsilniejszy sygnał empirical:

- `drug_to_burden`: AUPRC 0.719, Δ vs topology +0.441;
- `risk_modifier`: AUPRC 0.328, Δ +0.268.

### E4: relation ablation

30 runów: SAGE L2 × 5 seedów × full + 5 usunięć relacji.

- full: 0.640 validation AUPRC;
- bez `drug_to_burden`: −0.025;
- bez `interaction_amplify`: −0.022;
- bez `drug_to_load`: −0.019.

Decyzja Wave 1: cechy i topologia są potrzebne. Legacy baseline został
zamknięty; kolejna fala miała wybrać lepszy encoder.

## Wave 2 — screening architektur bez HCR

### Architecture screening

9 runów: HGT, GATv2 i matched SAGE × 3 seedy.

- HGT: valid 0.636, test 0.560;
- GATv2: 0.586 / 0.469;
- matched SAGE: 0.567 / 0.506.

### Depth

18 runów: HGT i SAGE × L1–L3 × 3 seedy.

- HGT: L1 0.645 > L2 0.634 > L3 0.618;
- SAGE: L1 0.597 > L2 0.575 > L3 0.564.

Wynik wspierał hipotezę oversmoothing: głębsze encodery nie pomagały.

### Heads i boundary

- HGT L1 H2: 0.638;
- H4: 0.657;
- H8: 0.668.

Kontrola H16 na 5 seedach dała 0.649 vs 0.655 dla H8,
`Δ(H16−H8)=−0.0056`.

### Final

10 runów: HGT L1 H8 vs matched SAGE L1 × 5 seedów.

- HGT: valid 0.6550±0.0057, test 0.5716, Brier 0.1603;
- SAGE: valid 0.6053±0.0213, test 0.4963, Brier 0.1723.

Decyzja: zamrożono HGT L1, H8, hidden 64, dropout 0.2, residual. Następna fala
miała sprawdzić HCR po stronie dekodera.

## Wave 3 — binary HCR

### Stage 1

Screen wariantów HCR:

- `binary_compact`: valid 0.7418, test 0.6884;
- `classical_binary`: 0.6871 / 0.5906;
- `binary_minimal`: 0.6809 / 0.6141;
- placebo `all_zero`: 0.6629 / 0.5725.

### Stage 2, 5 seedów

Paired `binary_compact` vs HGT bez HCR:

- mean Δvalid +0.0868±0.0171;
- 5/5 wygranych;
- mean Δtest +0.1168;
- mean Δvalid Brier −0.0326.

### Kontrole

- supported edges: Δ około +0.240;
- unsupported edges: Δ około +0.006;
- `all_zero−HCR0` około +0.008;
- `binary_compact−all_zero` około +0.079.

Decyzja: wzrost pochodził z treści HCR, nie tylko szerszego dekodera.
Zaakceptowano 8D `binary_compact`.

## Wave 3B / observed-gate motif completion

Ukrywano A→G, pozostawiano B→G i używano oracle co-parent B jako kontekstu.
Sześć wariantów × 3 seedy. Motif AUPRC:

- HCR3 bez a111: 0.6364±0.1384;
- HCR3 full: 0.5998±0.1924;
- HCR2 binary compact: 0.5231±0.2298;
- random context: 0.4274±0.0868;
- shuffled: 0.3973±0.1362;
- none: 0.1869±0.0100.

Decyzja: kontekst pomaga, ale oracle gate/co-parent trzeba zastąpić konstrukcją
latentną. Pełny `all_runs.csv` dla tej fali jest niekompletny; wyniki pochodzą
z plików `motif_metrics_*.json`.

## Wave 4 — latent HCR i wybór kontekstu

### Wave 4C: latent-gate recovery

Trzy seedy. Model nie mógł używać wartości ukrytej bramki G.

- latent pairwise AB⊕AY⊕BY, 24D: 0.6789 motif AUPRC;
- HCR3 bez a111: 0.5777;
- random context: 0.5180;
- HCR3 full: 0.4527;
- shuffled: 0.4368;
- none: 0.1700.

### Wave 4D: context-role audit

36 runów: 6 wariantów × 3 seedy × 2 maski.

Dla hide parent A:

- all-context top1: AUPRC 0.7267, ale precision@1=0;
- selected capacity-matched: 0.7028, precision@1=1;
- structural latent pairwise: 0.5157, precision@1=1 i FPR collider=0;
- structural shuffled: 0.3289;
- matched random: 0.3194;
- none: 0.1795.

Decyzja nie była oparta wyłącznie na AUPRC. Wybrano struktural latent pairwise,
ponieważ miał czystszy profil ról i nie wymagał oracle skanowania wszystkich
kolumn. To utworzyło tor L2 structural HCR.

## Wave 5 — ścieżki WNERW i link prediction

### Wave 5A: path-energy smoke

Bez treningu GNN. Na zamrożonych edge probabilities porównano uniform, HGT,
HCR2 i shuffled HCR w 5 seedach. Wczesne TPM było obciążone sposobem traktowania
unreachable queries, dlatego dodano Wave 5A.1.

### Wave 5A.1: common-query audit

Coverage prawdziwych ścieżek:

- train-only: 0.418;
- HGT: 0.510;
- HCR2: 0.649;
- patient shuffle/refit: 0.524.

Na wspólnych reachable queries paired TPM:

- HCR2 0.356;
- HGT 0.361;
- shuffle 0.362.

Wniosek: HCR zwiększa coverage grafu G*, ale nie daje unikalnej przewagi
energetycznej na wspólnych ścieżkach.

### Wave 5B: topology–energy decomposition

Panel T potwierdził coverage 0.418→0.510→0.649 dla train→HGT→HCR.

Panel E:

- uniform MRR@100 0.631;
- HCR2 0.602;
- HGT 0.587.

Zamrożono `temperature=0.5`, `length_penalty=0.1`,
`uncertainty_penalty=0`. Wynik B: topologia pomaga, energia nie wykazała
unikalnej przewagi.

### Wave 5C: personalizacja pacjenta

Pięć seedów, stałe G*. Aktywność pacjenta mogła używać tylko
`drug_exposure` i `patient_context`; endpointy i mechanizmy były wykluczone.

- structural gate: ΔHEM 0.473, consistency 1.0;
- patient shuffle: 0.457;
- context shuffle: 0.030;
- matched random: −0.017.

Decyzja: personalizacja mechanistyczna potwierdzona; gate weight=2.0.

### Wave 5D: path-supported link prediction

L0–L10, 5 seedów, clean. Dedykowany MLP krawędzi używał batchy po 256,
25 epok i selekcji po valid AUPRC.

Test AUPRC:

- L2 structural HCR: 0.734±0.012;
- L4 final z path support: 0.725±0.010;
- L0 HGT: 0.569;
- kontrola bez leave-one-out: 0.754, ale metodologicznie przeciekowa.

L4 przegrał z L2 o 0.0086 i wygrał tylko 2/5 seedów.

Decyzja: edge predictor = L2 structural HCR; WNERW pozostaje rerankerem ścieżek.
Wave 5E, czyli pełne scenariusze, była planowana, ale nie została uruchomiona.

## Wave 6 — mixed-type HCR feature engineering

Bez treningu GNN. Fit train-only, Legendre degree 4, ridge 0.01,
bootstrap 10.

- 229 motywów;
- 687 wierszy par;
- 560 par binary compact;
- 127 par mixed HCR.

Decyzja: pipeline cech mixed-type był gotowy do audytu architektury i GHCR.

## L2 Architecture Audit — pomost Wave 6→7

Stage 0 odtworzył L2 end-to-end: valid 0.775, test 0.776.
Stage 1: 21 konfiguracji × 3 seedy. Stage 2 planował 16×5=80 runów;
historyczna tabela programu była częściowa, ale finalny freeze istnieje.

Wybrano:

- HGT L3 H8 hidden 32, leaky ReLU, residual, dropout 0.2;
- decoder [256,128];
- valid 0.808, test 0.772;
- do 300 epok, patience 40.

## Wave 7 — GHCR i finalny decoder

### V0/V1/V2

Clean, jeden seed, 300 epok:

- V2 all-types: valid 0.768, test 0.633;
- V0 legacy compact: valid 0.753, test 0.822;
- V1 binary GHCR: valid 0.746, test 0.597.

Valid wskazywał V2, ale test i Brier wspierały V0. Kontrola V1B potwierdziła,
że samo a11 jest zgodne z φ V0, ale GHCR brakowało innych statystyk compact.

### Panel B: 40D na rolę

- B0 legacy pad40: valid 0.804, test 0.809;
- B2 enriched40: 0.791 / 0.822;
- B3 hybrid: 0.749 / 0.798;
- B1 matrix40: 0.702 / 0.702.

### Panel B residual

- R1 shared pair encoder: valid 0.876;
- R3 classical4 residual: 0.898;
- R4 + conditional edge gain: 0.892;
- R2 gated residual: 0.843.

Conditional edge gain nie pomógł: R4−R3≈−0.005.

### Wave 7C architecture audit

Clean, seed 20260722:

- A1 unshared role encoders: valid 0.922;
- A0 shared MLP: 0.918;
- A2 global motif: 0.863;
- A3 shared linear: 0.837.

Context ablations:

- AG only: 0.927;
- full: 0.919;
- no direct AG: 0.646;
- mask type/support: 0.674;
- patient permutation: 0.665.

AG było konieczne, ale pełny motyw zapewniał kontekst.

### Multi-seed freeze

- A1 unshared: 0.952±0.026 valid;
- shared+role masks: 0.949±0.033.

Zamrożono A1.

### Wave 7D: CMI i scenariusze

Conditional edge gain:

- T0 A1: valid 0.922, test 0.910;
- T1 + CMI: valid 0.916, test 0.956.

Ponieważ selekcja była po valid, CMI odrzucono.

T0 × 6 scenariuszy × 3 seedy:

- clean 0.952±0.026 valid;
- multihospital 0.964±0.027;
- no_overlap 0.956±0.036;
- hidden_confounder 0.952±0.026;
- selection_bias 0.942±0.039;
- noisy_documentation 0.952±0.024.

Finalny stack: HGT L3H8W32 + enriched 40D AZ⊕AG⊕ZG + unshared MLP.

## Wave 8 — brak

Nie ma folderów, skryptów, configów ani wyników Wave 8. Numeracja przechodzi
z Wave 7D do Wave 9. Nie należy sztucznie przypisywać Wave 7D numeru 8.

## Wave 9 — direct MLP vs KAN pair encoder

Clean, seed 20260722, 200 epok. Jedyną zmianą był pair encoder.

- MLP K0: valid 0.922789, test 0.914965, Brier 0.062489, 179 643 parametrów;
- KAN K1: valid 0.888078, test 0.930612, Brier 0.069010, 202 395 parametrów;
- Δvalid KAN−MLP = −0.034711.

Grid out-of-range był znikomy, więc adaptive grid nie był uzasadniony.
Zerowanie AG powodowało spadek około 0.253, wskazując AG jako kluczową rolę.

Decyzja: MLP. Dalszy full KAN multi-seed został zatrzymany.

## Wave 10 — AG-only KAN residual

Testowano:

`g_AG*=g_AG^MLP + sigmoid(a)·r_AG^KAN`,

z `a_init=−3`, czyli początkową bramką około 0.047.

Pierwszy retrain R0 był nieważny, bo nie odtwarzał checkpointu Wave 9.
W kanonicznym fair screen R0 był eval-only z Wave 9:

- R0 MLP: valid 0.922051, test 0.915079, Brier 0.064634;
- R1 residual: 0.922085, 0.915568, 0.064509;
- Δvalid +0.000034;
- finalna bramka α=0.0475, praktycznie bez otwarcia.

Decyzja: residual nie wniósł mierzalnej wartości; pozostaje MLP.

## Wave 11 — pełny retrain i audyt KAN

### Etap 0: direct MLP vs KAN

36 niezależnych runów: 2 modele × 6 scenariuszy × 3 seedy. Cały model był
trenowany od zera: HGT, pair encodery, graph branch i classifier.

Mean Δvalid `(KAN−MLP)`:

- clean −0.013442;
- hidden confounder −0.014610;
- selection bias −0.012899;
- no overlap −0.007028;
- noisy documentation −0.011688;
- multihospital −0.006889;
- średnia po scenariuszach około −0.0111.

Globalnym zwycięzcą pozostał MLP.

### Etap 1: K1–K5

Screen clean + multihospital, seed 20260722, 10 runów:

- K1 shallow: Δ −0.0028 / −0.0063;
- K2 MLP→KAN: −0.0034 / +0.0020;
- K3 KAN→Linear: −0.0320 / −0.0044;
- K4 Linear+KAN: −0.0960 / −0.1156;
- K5 grouped KAN: −0.0045 / −0.2482.

Promowano K1 i K2. K6–K9 miały tylko config stubs i nie były uruchomione.

### Etap 2: multi-seed K1/K2

12 runów: 2 architektury × 2 scenariusze × 3 seedy.

- K1: Δ vs MLP −0.002419 clean i −0.002767 multihospital;
- K2: −0.005395 i −0.002475.

Wybrano K1 shallow jako najlepszą architekturę KAN.

### Etap 3: K1 wszystkie scenariusze

18 runów. Mean Δvalid K1−MLP:

- no overlap +0.000803;
- clean −0.002419;
- hidden confounder −0.002596;
- multihospital −0.002767;
- noisy documentation −0.005048;
- selection bias −0.014591.

K1 wygrała 5 z 18 pojedynczych paired runów. Średnia po 18 runach:
−0.004436±0.008917. MLP pozostaje globalnym zwycięzcą.

### Audyt endpoint-path

Krawędź A→G była przypisywana do endpointu `e`, jeśli w zamrożonym downstream
grafie istniała ścieżka G⇝e. Kandydacka krawędź A→G nie była używana do
ustalania reachability.

Końcowy audyt rare endpoints opisuje `04_RARE_ENDPOINTS.md`.
