"""taskA.features — statystyczne cechy par (S0–S10) doklejane do krawędzi.

Publiczne API (czytaj to):
  variants.py   rejestr S0–S10; FINAL używa S10_HCR_FULL40 (40D)
  compute.py    wyliczanie wektorów na train patients
  attach.py     fit train-only → tensor stat_raw [N,3,D] + maski ról

Wnętrze (nie tunuj, jeśli odtwarzasz S10):
  pair_basis/hcr40/   bazy 40D pary (Legendre / discrete / packing)
  context/coparent/   rejestr koparentów Z z G_train (nigdy G_true)
  context/topology/   graf ze scored edges do budowy rejestru

Co wolno zmieniać:
  - wariant S0–S10 w experiment.stat_variant (NOWY eksperyment, nie FINAL)
  - nic w pair_basis/ jeśli chcesz te same 40 liczb co 14.08

Czego nie ruszać dla FINAL:
  - fit wyłącznie na pacjentach train
  - Z z edge-context registry, nie z true graph
"""
