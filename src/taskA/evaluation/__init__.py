"""taskA.evaluation — metryki po logitach krawędzi.

metrics.py         AUPRC, AUROC, Brier, F1, próg z validation
oversmoothing.py   diagnostyka wygładzania HGT
group_metrics.py   grupy krawędzi
endpoint_paths.py  ścieżki do endpointów (raport pathway)
per_edge_type.py   agregacje po typie krawędzi

Selekcja modelu FINAL: wyłącznie valid AUPRC.
"""

from taskA.evaluation.metrics import SynEvaluator

__all__ = ["SynEvaluator"]
