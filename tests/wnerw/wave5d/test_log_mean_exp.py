import torch

from link_prediction.wave5d.cohort_pooling import log_mean_exp_pool


def test_kappa_mean_like():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # small kappa → close to mean
    pooled = log_mean_exp_pool(x, kappa=1e-3, dim=0)
    assert torch.allclose(pooled, x.mean(dim=0), atol=1e-2)


def test_kappa_max_like():
    x = torch.tensor([[1.0, 0.0], [10.0, 2.0]])
    pooled = log_mean_exp_pool(x, kappa=50.0, dim=0)
    assert torch.allclose(pooled, x.max(dim=0).values, atol=0.2)
