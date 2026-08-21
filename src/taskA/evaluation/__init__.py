"""taskA.evaluation — metrics on edge logits.

metrics.py         AUPRC, AUROC, Brier, F1, validation threshold
oversmoothing.py   HGT oversmoothing diagnostics
group_metrics.py   edge groups
endpoint_paths.py  paths to endpoints (pathway report)
per_edge_type.py   aggregations by edge type

FINAL model selection: valid AUPRC only.
"""

from taskA.evaluation.metrics import SynEvaluator

__all__ = ["SynEvaluator"]
