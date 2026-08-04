#!/usr/bin/env python3
"""Smoke tests for Task A matched architecture family (no HCR).

Checks environment, metadata, shapes, finiteness, gradients, and a 3-epoch train.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from hydra import compose, initialize_config_dir
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from experiments.taskA_interventions import graph_fingerprint
from train_taskA import build_model, count_parameters, resolve_device, set_seed, train_epoch


MODELS = (
    "TaskA_hetero_sage_matched",
    "TaskA_hetero_gatv2",
    "TaskA_hgt",
)


def compose_cfg(model_name: str, epochs: int = 3):
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"model={model_name}",
                "data.dataset.scenario=clean",
                "data.feature_ablation_profile=empirical",
                "data.candidate_seed=20260722",
                "training.seed=20260721",
                f"training.epochs={epochs}",
                "wandb.enabled=false",
                "experiment.wave=ARCH_SMOKE",
            ],
        )


def smoke_one(model_name: str) -> None:
    print("=" * 72)
    print(f"SMOKE {model_name}")
    cfg = compose_cfg(model_name, epochs=3)
    set_seed(int(cfg.training.seed))
    device = resolve_device(cfg)
    train_data, valid_data, test_data, _ = load_recon_heterodata(cfg)
    expected_fp = graph_fingerprint(train_data)

    print("NODE TYPES:")
    for node_type in train_data.node_types:
        print(f"  {node_type}: {tuple(train_data[node_type].x.shape)}")
    print("EDGE TYPES:", len(train_data.edge_types))
    print("METADATA node_types:", len(train_data.metadata()[0]))
    print("METADATA edge_types:", len(train_data.metadata()[1]))

    model = build_model(cfg, train_data).to(device)
    train_data = train_data.to(device)

    logits = model(train_data)
    labels = train_data.edge_label.float().view(-1)
    assert logits.ndim == 1
    assert logits.numel() == labels.numel()
    assert torch.isfinite(logits).all()

    z_dict = model.encode(train_data)
    hidden = int(cfg.model.hidden_channels)
    for node_type in train_data.node_types:
        assert node_type in z_dict
        assert z_dict[node_type].shape == (
            train_data[node_type].num_nodes,
            hidden,
        )
        assert torch.isfinite(z_dict[node_type]).all(), node_type

    assert graph_fingerprint(train_data) == expected_fp

    n_pos = float(labels.sum().clamp(min=1.0))
    n_neg = float((labels.numel() - labels.sum()).clamp(min=1.0))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(n_neg / n_pos, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr))

    # Gradient check
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(train_data).view(-1), labels)
    loss.backward()
    grads = [
        p.grad.detach().norm().item()
        for _, p in model.encoder.named_parameters()
        if p.grad is not None
    ]
    assert grads, "No encoder gradients"
    assert max(grads) > 0, "Encoder gradients are all zero"
    optimizer.zero_grad(set_to_none=True)

    param_counts = count_parameters(model)
    print("params", param_counts)

    losses = []
    for epoch in range(1, 4):
        losses.append(train_epoch(model, train_data, optimizer, criterion, device))
        print(f"  epoch {epoch}: loss={losses[-1]:.4f}")
    assert all(torch.isfinite(torch.tensor(losses)))
    assert not (abs(losses[0] - losses[-1]) < 1e-12), "Loss stayed exactly constant"

    # Mini overfit probe on a balanced candidate subset (dropout off).
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    y_cpu = labels.detach().cpu().numpy()
    pos_idx = (y_cpu == 1).nonzero()[0][:32]
    neg_idx = (y_cpu == 0).nonzero()[0][:32]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise RuntimeError("Cannot overfit-probe: missing class in train candidates")

    idx = torch.as_tensor(
        np.concatenate([pos_idx, neg_idx]),
        device=device,
        dtype=torch.long,
    )
    probe_opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        probe_opt.zero_grad(set_to_none=True)
        full_logits = model(train_data).view(-1)
        probe_loss = criterion(full_logits[idx], labels[idx])
        probe_loss.backward()
        probe_opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(model(train_data).view(-1)[idx]).cpu().numpy()
        y = labels[idx].cpu().numpy()
        auprc = float(average_precision_score(y, probs))
    print(f"  mini-overfit AUPRC@{int(idx.numel())}: {auprc:.4f}")
    assert auprc > 0.90, f"Mini-overfit failed for {model_name}: AUPRC={auprc}"

    # Keep splits unused but referenced so loaders stay consistent.
    _ = valid_data, test_data
    print(f"SMOKE OK: {model_name}")


def main() -> None:
    print("ENV AUDIT")
    print("PyTorch:", torch.__version__)
    print("PyG:", torch_geometric.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print(
        "MPS available:",
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    )
    for model_name in MODELS:
        smoke_one(model_name)
    print("\nALL ARCHITECTURE SMOKES PASSED")


if __name__ == "__main__":
    main()
