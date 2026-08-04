# Wzory, metryki i agregacja statystyczna

## 1. Notacja

- `G_true` — pełny audytowany graf prawdy;
- `G_train` — graf message passingu z dodatnich krawędzi train;
- `s,t` — source i target kandydackiej krawędzi;
- `y_i∈{0,1}` — prawdziwa etykieta kandydata `i`;
- `l_i∈R` — logit modelu;
- `p_i=σ(l_i)` — przewidywane prawdopodobieństwo;
- `N` — liczba ocenianych kandydatów;
- `N_+` i `N_-` — liczby pozytywów i negatywów;
- `S` — liczba seedów;
- `m_s^A` — wartość metryki modelu A na seedzie `s`.

## 2. Prawdopodobieństwo krawędzi

Model zwraca logit `l_i`. Prawdopodobieństwo:

`p_i = σ(l_i) = 1 / (1 + exp(−l_i))`.

`σ` jest funkcją sigmoid, a `exp` funkcją wykładniczą.

## 3. Ważona binary cross-entropy

`L_BCE = −(1/N) Σ_i [w_+ y_i log(p_i) + (1−y_i)log(1−p_i)]`,

gdzie:

- `w_+ = N_-/N_+` jest wagą klasy pozytywnej;
- `y_i=1` dla prawdziwej krawędzi;
- `p_i` jest prawdopodobieństwem modelu.

W KAN opcjonalnie dodawano karę spline:

`L = L_BCE + λ_spline Σ_j ||c_j||_1`,

gdzie `c_j` oznacza współczynniki spline, `||·||_1` sumę wartości
bezwzględnych, a `λ_spline=10^−5`.

## 4. Wektor pary i decoder

Dla embeddingów źródła i celu:

`q_st = z_s ⊕ z_t ⊕ (z_s⊙z_t) ⊕ |z_s−z_t|`,

gdzie:

- `z_s,z_t` — embeddingi węzłów z encodera GNN;
- `⊕` — konkatenacja;
- `⊙` — iloczyn element po elemencie;
- `|·|` — wartość bezwzględna element po elemencie.

W finalnym modelu:

`g_HCR = E_AZ(h_AZ) ⊕ E_AG(h_AG) ⊕ E_ZG(h_ZG)`,

`g_final = g_graph ⊕ g_HCR`,

`l_st = f_classifier(g_final)`.

`E_role` oznacza encoder jednej roli HCR, a `f_classifier` końcowy MLP.

## 5. KANLinear

Pojedyncze połączenie KAN można zapisać:

`φ_ji(x_i) = w_ji^base SiLU(x_i) + w_ji^spline Σ_m c_jim B_m(x_i)`,

`o_j = Σ_i φ_ji(x_i) + b_j`,

gdzie:

- `x_i` — i-ty kanał wejścia;
- `o_j` — j-ty kanał wyjścia;
- `SiLU(x)=xσ(x)` — bazowa aktywacja;
- `B_m` — m-ta funkcja bazowa B-spline;
- `c_jim` — współczynnik spline;
- `w^base`, `w^spline` — skale części bazowej i spline;
- `b_j` — bias.

## 6. Prevalence i baseline AUPRC

`π = N_+ / N`.

`π` jest częstością klasy pozytywnej w ocenianym zbiorze i baseline AUPRC dla
losowego rankingu.

## 7. Average Precision / AUPRC

W implementacji używano `average_precision_score`. Dyskretna postać:

`AP = Σ_k (R_k − R_{k−1}) P_k`,

gdzie:

- `k` indeksuje kolejne progi rankingu;
- `P_k` jest precision po zastosowaniu progu `k`;
- `R_k` jest recall po zastosowaniu progu `k`.

Precision i recall:

`Precision = TP/(TP+FP)`,

`Recall = TP/(TP+FN)`.

`TP`, `FP`, `FN` oznaczają odpowiednio true positives, false positives i
false negatives.

## 8. AUROC

Interpretacja probabilistyczna:

`AUROC = P(p_positive > p_negative)`,

czyli prawdopodobieństwo, że losowy pozytyw uzyska wyższy score niż losowy
negatyw. Przy remisach stosuje się standardową korektę 0.5.

## 9. Brier score

`Brier = (1/N) Σ_i (p_i−y_i)^2`.

Niższa wartość oznacza lepszą kalibrację i trafność probabilistyczną.

## 10. F1 i F2

`F_β = (1+β²)·Precision·Recall / (β²·Precision + Recall)`.

- dla `β=1` otrzymujemy F1;
- dla `β=2` otrzymujemy F2, które silniej waży recall.

Próg klasyfikacji wybierano na validation jako próg maksymalizujący F1,
a następnie stosowano ten sam próg do finalnego train/valid/test.

## 11. Lift AUPRC

`Lift = AUPRC / π`.

`Lift=1` oznacza wynik równy losowemu baseline. `Lift>1` oznacza poprawę
względem częstości pozytywów.

## 12. Normalized AUPRC

`nAUPRC = (AUPRC−π)/(1−π)`.

Znaczenie:

- `0` — wynik równy prevalence baseline;
- `1` — perfekcyjny ranking;
- wartość ujemna — wynik poniżej baseline.

Metryka ułatwia porównanie endpointów o różnej prevalence.

## 13. Średnia i sample standard deviation

`mean(m) = (1/S) Σ_s m_s`,

`SD(m) = sqrt[(1/(S−1)) Σ_s (m_s−mean(m))²]`.

`S` jest liczbą seedów. W raportach `mean±SD` pokazuje centralny wynik i
zmienność inicjalizacji/treningu.

## 14. Paired delta

Dla modeli A i B na identycznym seedzie i fingerprintach:

`Δ_s = m_s^A − m_s^B`.

Średnia różnica:

`mean(Δ) = (1/S) Σ_s Δ_s`.

W Wave 11 przyjmowano:

`Δ_scenario = mean_seed(AUPRC_KAN − AUPRC_MLP)`.

Paired delta jest uczciwsza niż różnica dwóch niezależnych średnich, ponieważ
kontroluje wspólny split i seed.

## 15. Liczba wygranych

`wins_A = Σ_s 1[Δ_s>0]`,

gdzie `1[warunek]` jest funkcją wskaźnikową równą 1, gdy warunek zachodzi.

Przykład Wave 3: binary compact wygrał 5/5 seedów. K1 shallow w Wave 11
wygrał 5/18 paired runów globalnych.

## 16. Bootstrap confidence interval

W notebooku raportowym stosowano 10 000 bootstrap resamplings paired deltas:

1. losuj ze zwracaniem `S` wartości z `{Δ_1,...,Δ_S}`;
2. policz średnią bootstrapową `mean(Δ*)`;
3. powtórz `B=10 000` razy;
4. CI95% = percentyle 2.5% i 97.5% rozkładu bootstrapowych średnich.

`Δ*` oznacza próbę bootstrapową. CI było diagnostyką; główny pipeline nie
stosował formalnych testów t ani Wilcoxona.

## 17. HCR dla par binarnych

Niech:

- `p11=P(U=1,V=1)`;
- `p10=P(U=1,V=0)`;
- `p01=P(U=0,V=1)`;
- `p00=P(U=0,V=0)`;
- `pU=P(U=1)`;
- `pV=P(V=1)`.

Conditional probability:

`CP = P(V=1|U=1) = p11/pU`.

Risk difference:

`RD = P(V=1|U=1) − P(V=1|U=0)`.

Log odds ratio, ze stabilizacją `ε`:

`LOR = log[((p11+ε)(p00+ε))/((p10+ε)(p01+ε))]`.

Phi:

`φ = (p11 p00 − p10 p01) / sqrt[(p11+p10)(p01+p00)(p11+p01)(p10+p00)]`.

Mutual information:

`MI(U,V) = Σ_u Σ_v p(u,v) log[p(u,v)/(p(u)p(v))]`.

Support:

`support = n_complete/n_train`.

## 18. Path coverage

`Coverage = (1/Q) Σ_q 1[Γ_q^true ∩ Ω_q ≠ ∅]`,

gdzie:

- `Q` — liczba zapytań source→endpoint;
- `Γ_q^true` — zbiór prawdziwych ścieżek dla query `q`;
- `Ω_q` — zbiór ścieżek osiągalnych w analizowanym `G*`.

## 19. True Path Mass

`TPM_q = Σ_{γ∈Γ_q^true} P(γ|q)`,

`TPM = (1/Q) Σ_q TPM_q`.

`γ` oznacza ścieżkę, a `P(γ|q)` znormalizowaną masę/energię ścieżki w query.
W poprawionym protokole `TPM_q=0`, jeśli prawdziwa ścieżka nie jest osiągalna.

## 20. MRR i Hits@K

`RR_q = 1/rank_q`,

`MRR = (1/Q) Σ_q RR_q`,

gdzie `rank_q` jest rangą pierwszej prawdziwej ścieżki.

`Hits@K = (1/Q) Σ_q 1[rank_q≤K]`.

## 21. Energia ścieżki WNERW

Uproszczona postać używana do interpretacji:

`E(γ) = −(1/T) Σ_{e∈γ} log p_e + λ_L |γ| + λ_U U(γ) − η A_patient(γ)`,

gdzie:

- `T` — temperatura;
- `p_e` — prawdopodobieństwo krawędzi;
- `|γ|` — długość ścieżki;
- `λ_L` — kara długości;
- `U(γ)` — niepewność ścieżki;
- `λ_U` — waga niepewności;
- `A_patient(γ)` — aktywacja zgodna z profilem pacjenta;
- `η` — waga bramki pacjenta.

Wave 5B zamroziła `T=0.5`, `λ_L=0.1`, `λ_U=0`; Wave 5C `η=2.0`.

## 22. Endpoint reachability

Dla kandydata A→G i endpointu `e`:

`r(A→G,e) = 1`, jeśli w zamrożonym downstream grafie istnieje `G⇝e`.

Kandydacka krawędź A→G nie jest dodawana do grafu podczas ustalania `r`.
Jedna krawędź może mieć `r=1` dla wielu endpointów.

Średnia shortest path:

`mean_length_e = (1/N_e) Σ_i d(G_i,e)`,

gdzie `d(G_i,e)` jest długością najkrótszej skierowanej ścieżki od targetu
kandydata do endpointu.

## 23. Reguły decyzyjne

Decyzje były predefiniowanymi progami, nie testami p-value. Przykłady:

- Wave 3: mean Δvalid ≥0.05, co najmniej 4/5 wygranych, SD≤0.05 i Brier nie
  gorszy;
- Wave 9/10: promocja KAN wymagała około +0.005 validation AUPRC;
- Wave 11 K1–K5: promocja screen, gdy mean delta ≥−0.005 i worst delta ≥−0.05;
- test pozostawał report-only i nie zmieniał decyzji checkpointu.
