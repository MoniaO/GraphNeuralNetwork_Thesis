"""Task A — rekonstrukcja skierowanych krawędzi na GSN v3.

Układ pakietu (czytaj od góry):

  taskA.data           łączenie grafu, pacjentów i kandydatów
  taskA.features       cechy S10 (40D × AZ/AG/ZG), attach train-only
  taskA.models.encoder HGT (FINAL) oraz SAGE/GAT/RGCN (Stage A)
  taskA.models.decoder Fusion88 + MLP/KAN pair encoder
  taskA.training       pętla train / early stop / test sealed
  taskA.evaluation     AUPRC i metryki pomocnicze
  taskA.experiments    Stage A → Stage C → FINAL 14.08

Zamrożony stack FINAL: HGT h32 L2 d0.25 heads=4 + fusion88_stat + S10.
Nie zmieniaj hypers z `experiments.final_14_08` jeśli chcesz odtworzyć tabelę 14.08.
"""
