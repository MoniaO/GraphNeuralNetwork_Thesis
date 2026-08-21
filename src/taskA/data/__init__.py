"""taskA.data — budowa grafu GSN v3 i zbiorów kandydatów.

Co tu jest:
  load_graph.py       główny loader HeteroData (train/valid/test)
  hetero_graph.py     native PyG HeteroData z CSV
  patient_matrix.py   pacjenci + split; fit cech tylko na train
  candidate_pairs.py  pary (u,v) z HeteroData
  layout.py           globalny porządek węzłów (encoder ↔ decoder)
  feature_ablation.py profil cech węzła (empirical / topology_only)
  interventions.py    kontrolowane usuwanie relacji
  raw/                generatory syntetycznego GSN v3 (nie część treningu)

Co wolno zmieniać przy nowych eksperymentach:
  - scenario i candidate_seed w configs/data/dataset_v3.yaml
  - feature_ablation_profile (domyślnie empirical)

Czego nie ruszać dla odtworzenia FINAL 14.08:
  - candidate_seed=20260722
  - G_train tylko z dodatnich krawędzi train
  - brak G_true w message passingu
"""
