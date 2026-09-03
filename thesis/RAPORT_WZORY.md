# Raport: formatowanie LaTeX vs. szkic i skrypty

Rozdziały w `thesis/` to **formatowanie szkicu**, nie redakcja. Proza (w tym literówki) została. Wzory, które w wklejce były posklejane przez markdown, złożyłem tak, jak liczy je kod. Poniżej: co jest OK, co poprawiłem tylko w matematycznym składzie, i co zostawiłem w tekście, a wymaga Twojej decyzji.

Źródła wzorów: `src/taskA/features/pair_basis/hcr40/{packing,summaries,encoder,continuous_basis,jitter}.py`, `src/taskA/data/load_graph.py`, `src/taskA/models/decoder/{pair_encoder,fusion88}.py`, `src/taskA/training/train.py`. Rejestr zmiennych: `VARIABLE_SPECS` — 137 binary, 10 count, 8 continuous (zgodne ze szkicem).

---

## 1. Wzory złożone według kodu (markdown był nieczytelny)

Te miejsca **nie są zmianą Twojej myśli**, tylko odwróconym ułamkiem / urwany wzór w wklejce. W PDF-ie jest wersja ze skryptów.

| Miejsce | W szkicu (posklejane) | W kodzie i w LaTeX |
|---|---|---|
| Negatywy | `u=v` jako warunek | `source != target`, czyli \(u\neq v\) (`load_graph.py`) |
| \(\phi\) | licznik/mianownik zlane w jedną linię | standardowy Pearson \(\phi\) z tablicy \(2\times 2\); dla pary binarnej \(a_{11}\) w GHCR to ta sama wielkość w postaci \((p_{11}-p_U p_V)/\sqrt{\cdots}\) |
| \(D_{\mathrm{mean}}\) | wyglądało na \(d_U d_V \lVert A\rVert_F^2\) | \(\lVert A\rVert_F^2/(d_U d_V)\) (`dependence_summaries`) |
| \(R_{\mathrm{low}}\) | ułamek odwrócony | \(\lVert A_{:2,:2}\rVert_F^2/(\lVert A\rVert_F^2+\varepsilon)\) |
| \(M_{\max}\) | urwany `\displaystyle max a_{jk}` | \(\max_{j,k}\lvert a_{jk}\rvert\) — w kodzie jest **wartość bezwzględna** |
| \(S\) | wyglądało na \(n_{\mathrm{train}}/n_{\mathrm{complete}}\) | \(n_{\mathrm{complete}}/n_{\mathrm{train}}\) (slot 32) |
| ECDF \(u_i\) | `clip(N+1 , 1/2(ℓ+r)+0.5)` zlane | \((0.5(\ell_i+r_i)+0.5)/(N_{\mathrm{train}}+1)\), potem clip do \([10^{-6},1-10^{-6}]\) |
| Count, slot 2 | mianownik/licznik zlane | \(\mathbb{E}[\log(1+C^c)]/\log(1+c_{\max})\) |
| Count, slot 4 | \(\tanh[\log \mathbb{E}[C]/(\mathrm{Var}+\varepsilon)]\) | \(\tanh\log[(\mathrm{Var}+\varepsilon)/(\mathbb{E}[C]+\varepsilon)]\) — **odwrotny stosunek** niż w wklejce |
| Continuous, slot 3 | urwany `P(z_r` | \(P(\lvert z_r\rvert>1.5)\) |
| Binary entropy \(H_2\) | `ln 2` wyglądało jak w liczniku | \([-p\ln p-(1-p)\ln(1-p)]/\ln 2\) |
| Log-lift \(L\) | mianownik/licznik zlane | \(\log[(p_{++}+\varepsilon)/(p_+(U)p_+(V)+\varepsilon)]\) |
| \(w_+\) | raz \(N_{\mathrm{pos}}/N_{\mathrm{neg}}\), przykład 300/100=3 | `pos_weight = n_neg / n_pos` |
| Precision / Recall | `TP+FP` w liczniku | \(\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})\), analogicznie recall |
| Macierz \(A_{\mathrm{pad}}\) | pierwsza „wiersz” to \(a_{11},a_{21},\ldots\) (transpozycja) | packing `reshape(..., order="C")` = wierszami: \(a_{11},a_{12},\ldots\); złożyłem standardowy układ zgodny z \(\mathrm{slot}(a_{jk})=4(j-1)+(k-1)\) |

Jeśli chcesz, żebym **przywrócił wadliwe ułamki dosłownie ze szkicu**, napisz — wtedy PDF będzie zgadzał się z wklejką, a nie z kodem.

---

## 2. Zgodne ze skryptami (OK)

- \(a_{jk}=n^{-1}\sum_i \phi_j(U_i)\phi_k(V_i)\) i \(A=n^{-1}\Phi_U^\top\Phi_V\) — `encoder._coefficient_matrix`.
- Padding \(4\times 4\), 16 pierwszych slotów, B2 = FULL40.
- Baza binarna: jeden kontrast \((B-p)/\sqrt{p(1-p)}\); para binarna → blok \(1\times 1\) = Pearson \(\phi\).
- Continuous: train ECDF + przesunięte Legendre \(\sqrt{2j+1}\,P_j(2u-1)\), \(j=1,\ldots,4\).
- Count: cap \(q_{0.995}\), jitter \(u=F(c^{-})+\xi P(C=c)\), potem ta sama baza Legendre (nie binaryzacja).
- Marginesy 4D, joint activity, rarity \((n_{++}+1)^{-1/2}\), one-hot typów 3+3.
- Unsupported: \(n_{\mathrm{complete}}<50\); dla binary–binary min. 5 obserwacji na poziom.
- Motyw \(AZ\oplus AG\oplus ZG\) → \(\mathbb{R}^{120}\); trzy niezależne \(40\to 16\to 8\) → \(g_{\mathrm{stat}}\in\mathbb{R}^{24}\).
- \(q_{AG}=[z_A\parallel z_G\parallel z_A\odot z_G\parallel\lvert z_A-z_G\rvert]\), MLP \(4d\to 128\to 64\), fusion \(64+24=88\to 64\to 1\).
- MLP: Linear→GELU→LN→Dropout→Linear→LN. KAN: `KANLinear(40→8)` + LayerNorm.
- \(r_{\mathrm{neg}}=3\); allowed type pairs z dodatnich krawędzi, nie pełny iloczyn kartezjański typów.
- 137 / 10 / 8 zgadza się z zamrożonym rejestrem.

Drobiazg implementacyjny, nie trzeba zmieniać tekstu: BCE w PyTorch to `BCEWithLogitsLoss(pos_weight=...)` na logitach, nie sigmoid + BCE na prawdopodobieństwach. Wzór w rozdziale jest równoważny dydaktycznie.

Continuous slot 2: w kodzie `kurt = E[z0^4]-3`, potem `tanh(kurt/10)`. Twój zapis \(\tanh((\gamma_2-3)/10)\) przy \(\gamma_2=E[z_0^4]\) jest ten sam.

---

## 3. Proza zostawiona celowo (literówki / urwane zdania)

Nie ruszałam. Do Twojej ręcznej korekty:

- *Knoledge Graph*; *There fore The main*; brak kropki przed „model selection…” — **poprawione w passie redakcyjnej 22.08**
- *real word*, *methodhologcalyy calcluared assotianos* — **poprawione**
- *neigberhoud*, *whats should hepl*, *ftiig*, *knowledge egraph* — **poprawione**
- *This gives as very elegant conceptual split* — **poprawione**
- MLP vs KAN: „**lusion** is:” — **poprawione** (*The conclusion is*)
- Nagłówek rozdziału benchmark: **„Then describe three stages.”** — **usunięte** (22.08)
- W §3 warunek negatywów w szkicu miał `u=v`; w LaTeX jest \(u\neq v\) (pkt 1). Jeśli chcesz zostawić dosłownie `u=v`, trzeba świadomie cofnąć.

---

## 4. Liczby (szkic ≈ vs. tabele z treningu)

Szkic ma zaokrąglenia. Nie wstawiałam starych polskich tabel z `outputs/`, żeby nie dokładać tekstu spoza szkicu.

| Szkic | Artefakt Stage C (`STAGE_C_CLEAN_S0_S10_TABLE`) |
|---|---|
| HGT only \(\approx 0.719\) | S0: \(0.7186\pm 0.0147\) (to **Stage C** z zerowym \(g_{\mathrm{stat}}\), nie Top-1 Stage A) |
| \(\phi\) \(\approx 0.840\) | S4 binary: \(0.8397\) |
| NMI \(\approx 0.853\) | S1 NMI: \(0.8525\) |
| enriched \(\approx 0.913\) | S9 compact 36D: \(0.9125\pm 0.0114\) |
| S10 \(\approx 0.917\) | S10 FULL40: \(0.9167\pm 0.0006\) |
| makro \(\approx 0.915\) | S10 makro valid \(\approx 0.915\) — OK |

Uwaga metodologiczna: w Stage A sam HGT (matched freeze) miał valid AUPRC \(\approx 0.729\), a w §9.2 „HGT only \(\approx 0.719\)” to S0 z Stage C. Różnica protokołu (seedy / ten sam dekoder). Warto to potem dopisać jednym zdaniem, żebym nie mieszała backbone freeze z ablacją statystyk.

Ablacja 40 vs 36, \(\Delta\approx 0.007\): zgodne z Stage 1 na `clean` (średnia \(\approx -0.0065\)). Nie aktualizowałam o Stage 2.

Kolejność w szkicu: \(\phi\) (0.840) potem NMI (0.853). W tabeli Stage C NMI (S1) jest wyżej niż binary \(\phi\) (S4), ale S1 nie jest „po” S4 w numeracji wariantów — to kolejność narracyjna, nie kolejność eksperymentu.

---

## 5. Czego nie wstawiłam

- Komentarze redakcyjne ze szkicu („I would put that paragraph…”, „Recommended figures…”, mostek „That is the clean bridge…”).
- Zdanie zamykające podsekcję HCR **jest** w rozdziale (to, które chciałaś jako close).
- Akapit o 36D jest na końcu podsekcji statystycznej (Methods), nie w Results.
- Sześć figur architektury: tymczasowe schematy tekstowe w ramce (`fig:taskA-problem`, `motif`, `architecture`, `lastfm-problem`, `context`, `architecture`) — do podmiany finalnymi PDF-ami.
- `chapters/abstract.tex` + `chapters/streszczenie.tex` w `main.tex` (22.08).
- Stare tabele Stage A / FINAL 14.08 z poprzedniego szkieletu — mogę je wrzucić jako Table z `outputs/` gdy powiesz.

---

## 6. Kompilacja

`pdflatex` nie jest w PATH (MacTeX był pobrany, ale nie zainstalowany albo binarka nie jest na ścieżce). Po instalacji:

```bash
cd thesis && pdflatex main.tex && pdflatex main.tex
```

Pliki: `thesis/main.tex`, `thesis/chapters/taskA_{rozwiazanie,ewaluacja,benchmark}.tex`.
