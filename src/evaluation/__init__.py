from evaluation.ogb_evaluator import OGBEvaluator

def build_evaluator(cfg):
    evaluators = {
        "ogb": OGBEvaluator,
    }
    if cfg.data.evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator: {cfg.data.evaluator_type}")
    return evaluators[cfg.data.evaluator_type](cfg)