# Thesis — Task A + Last-FM* proxy

English thesis draft. Formulas and numbers follow `src/taskA/` and the Last-FM pipeline; prose discrepancies are tracked in `RAPORT_WZORY.md`.

| File | Content |
|---|---|
| `chapters/abstract.tex` | Abstract and keywords |
| `chapters/streszczenie.tex` | Polish summary (streszczenie) |
| `chapters/introduction.tex` | Hypothesis, mapping table, chapter roadmap |
| `chapters/taskA_rozwiazanie.tex` | Task~A: overview, data, HCR-inspired representation, motif, HGT, fusion |
| `chapters/taskA_ewaluacja.tex` | Weighted BCE, AUPRC protocol |
| `chapters/taskA_benchmark.tex` | Stage A / C / MLP vs KAN, results figures |
| `chapters/lastfm_proxy_rozwiazanie.tex` | Proxy: overview, A5/A11/H3/LEG, HGT, fusion |
| `chapters/lastfm_proxy_ewaluacja.tex` | BCE; sampled vs full-catalog NDCG |
| `chapters/lastfm_proxy_benchmark.tex` | Ablation, TRUE FINAL full-rank, discussion |
| `chapters/summary.tex` | Conclusions |

Architecture overview figures use inline text pipelines until final PDFs replace them.
Result figures for Task~A: `figures/fig_final_*.pdf` (regenerate with `thesis/scripts/make_taskA_figures.py`).

Compile (requires BibTeX):

```bash
cd thesis && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Bibliography: `references.bib` (HCR sources: Duda 2018a/b, Duda \& Szulc 2020; expository Wolfram Community overview).
