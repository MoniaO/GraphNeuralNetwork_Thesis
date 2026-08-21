"""taskA.features — pair statistics (S0–S10) attached to edges.

Public API (read this):
  variants.py   S0–S10 registry; FINAL uses S10_HCR_FULL40 (40D)
  compute.py    vector computation on train patients
  attach.py     train-only fit → stat_raw [N,3,D] + role masks

Internals (do not tune if you reproduce S10):
  pair_basis/hcr40/   40D pair bases (Legendre / discrete / packing)
  context/coparent/   coparent-Z registry from G_train (never G_true)
  context/topology/   scored-edge graph used to build the registry

What you may change:
  - S0–S10 variant in experiment.stat_variant (NEW experiment, not FINAL)
  - nothing in pair_basis/ if you want the same 40 numbers as 14.08

What not to touch for FINAL:
  - fit on train patients only
  - Z from the edge-context registry, not from the true graph
"""
