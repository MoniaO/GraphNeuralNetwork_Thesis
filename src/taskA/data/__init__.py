"""taskA.data — GSN v3 graph and candidate sets.

Contents:
  load_graph.py       main HeteroData loader (train/valid/test)
  hetero_graph.py     native PyG HeteroData from CSV
  patient_matrix.py   patients + split; feature fit on train only
  candidate_pairs.py  (u, v) pairs from HeteroData
  layout.py           global node order (encoder ↔ decoder)
  feature_ablation.py node-feature profile (empirical / topology_only)
  interventions.py    controlled relation removal
  raw/                GSN v3 generators (not part of training)

What you may change in a new experiment:
  - scenario and candidate_seed in configs/data/dataset_v3.yaml
  - feature_ablation_profile (default empirical)

What not to touch to reproduce FINAL 14.08:
  - candidate_seed=20260722
  - G_train from positive train edges only
  - no G_true in message passing
"""
