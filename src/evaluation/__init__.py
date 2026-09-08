# from omegaconf import OmegaConf

# from evaluation.ogb_evaluator import OGBEvaluator
# from evaluation.syntetic_evaluator_node import SynEvaluator


# def build_evaluator(cfg):
#     evaluator_type = OmegaConf.select(cfg, "data.evaluator_type")
#     if evaluator_type is None:
#         evaluator_type = OmegaConf.select(cfg, "evaluator_type")

#     if evaluator_type is None:
#         raise ValueError(
#             "Missing evaluator type in Hydra config. Set data.evaluator_type or evaluator_type."
#         )

#     evaluators = {
#         "ogb": OGBEvaluator,
#         "synthetic_evaluator": SynEvaluator
#     }

#     if evaluator_type not in evaluators:
#         raise ValueError(f"Unknown evaluator: {evaluator_type}")

#     return evaluators[evaluator_type](cfg)