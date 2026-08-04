"""Train / evaluate Wave 5D final edge decoder with cohort pooling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from link_prediction.wave5d.calibration import Calibrator, fit_temperature_bias
from link_prediction.wave5d.cohort_pooling import log_mean_exp_pool
from link_prediction.wave5d.final_edge_decoder import (
    FinalEdgeDecoder,
    VariantFeatureMask,
    apply_feature_mask,
    feature_dim,
)
from link_prediction.wave5d.patient_edge_features import EdgeFeatureBundle
from link_prediction.wave5d.role_metrics import (
    edge_classification_metrics,
    hits_at_k,
    role_specific_fpr,
)


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_edges: int = 256
    kappa: float = 1.0
    seed: int = 0
    device: str = "cpu"


def _split_index(bundle: EdgeFeatureBundle, split: str) -> np.ndarray:
    return np.array(
        [i for i, s in enumerate(bundle.splits) if str(s).lower() == split],
        dtype=np.int64,
    )


def _patient_index(bundle: EdgeFeatureBundle, split: str | None) -> np.ndarray:
    if split is None:
        return np.arange(len(bundle.patient_ids), dtype=np.int64)
    # For edge train use train patients; for valid/test eval use matching patients.
    return np.array(
        [
            i
            for i, s in enumerate(bundle.patient_splits)
            if str(s).lower() == split
        ],
        dtype=np.int64,
    )


def _masked_features(
    bundle: EdgeFeatureBundle,
    edge_idx: np.ndarray,
    patient_idx: np.ndarray,
    mask: VariantFeatureMask,
    device: torch.device,
) -> torch.Tensor:
    """Return [P, E, F] features for selected patients/edges."""
    hgt = torch.as_tensor(bundle.hgt_pair[edge_idx], device=device)  # [E, 4H]
    hcr = torch.as_tensor(bundle.hcr24[edge_idx], device=device)
    supp = torch.as_tensor(bundle.support[edge_idx], device=device)
    unc = torch.as_tensor(bundle.uncertainty[edge_idx], device=device)
    path = torch.as_tensor(
        bundle.path_completion[np.ix_(patient_idx, edge_idx)], device=device
    )
    gate = torch.as_tensor(
        bundle.gate[np.ix_(patient_idx, edge_idx)], device=device
    )

    # Broadcast static features across patients
    p = path.shape[0]
    hgt_b = hgt.unsqueeze(0).expand(p, -1, -1)
    hcr_b = hcr.unsqueeze(0).expand(p, -1, -1)
    supp_b = supp.unsqueeze(0).expand(p, -1)
    unc_b = unc.unsqueeze(0).expand(p, -1)

    return apply_feature_mask(
        hgt_pair=hgt_b,
        hcr24=hcr_b,
        path_completion=path,
        gate=gate,
        support=supp_b,
        uncertainty=unc_b,
        mask=mask,
    )


@torch.no_grad()
def predict_cohort_logits(
    model: FinalEdgeDecoder,
    bundle: EdgeFeatureBundle,
    edge_idx: np.ndarray,
    mask: VariantFeatureMask,
    kappa: float,
    device: torch.device,
    patient_split: str | None = "train",
) -> np.ndarray:
    model.eval()
    pidx = _patient_index(bundle, patient_split)
    if len(pidx) == 0:
        pidx = np.arange(len(bundle.patient_ids), dtype=np.int64)
    feats = _masked_features(bundle, edge_idx, pidx, mask, device)
    # feats: [P, E, F]
    p, e, f = feats.shape
    logits = model(feats.reshape(p * e, f)).reshape(p, e)
    pooled = log_mean_exp_pool(logits, kappa=kappa, dim=0)
    return pooled.cpu().numpy()


def train_variant(
    bundle: EdgeFeatureBundle,
    mask: VariantFeatureMask,
    cfg: TrainConfig,
    *,
    log_epoch_fn=None,
) -> tuple[FinalEdgeDecoder, Calibrator, dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    in_dim = feature_dim(bundle.hgt_pair.shape[1], mask)
    model = FinalEdgeDecoder(in_dim, cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_e = _split_index(bundle, "train")
    valid_e = _split_index(bundle, "valid")
    train_p = _patient_index(bundle, "train")
    if len(train_p) == 0:
        train_p = np.arange(len(bundle.patient_ids), dtype=np.int64)

    best_state = None
    best_valid = -1.0
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        order = np.random.permutation(train_e)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), cfg.batch_edges):
            batch = order[start : start + cfg.batch_edges]
            # Balanced sampling within batch
            y = bundle.labels[batch]
            pos = batch[y >= 0.5]
            neg = batch[y < 0.5]
            if len(pos) and len(neg):
                n = min(len(pos), len(neg), cfg.batch_edges // 2)
                batch = np.concatenate(
                    [
                        np.random.choice(pos, n, replace=len(pos) < n),
                        np.random.choice(neg, n, replace=len(neg) < n),
                    ]
                )
            feats = _masked_features(bundle, batch, train_p, mask, device)
            p, e, f = feats.shape
            logits = model(feats.reshape(p * e, f)).reshape(p, e)
            pooled = log_mean_exp_pool(logits, kappa=cfg.kappa, dim=0)
            labels = torch.as_tensor(
                bundle.labels[batch], device=device, dtype=torch.float32
            )
            loss = nn.functional.binary_cross_entropy_with_logits(pooled, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        # Valid AUPRC for early selection
        v_logits = predict_cohort_logits(
            model, bundle, valid_e, mask, cfg.kappa, device, patient_split="valid"
        )
        from link_prediction.wave5d.role_metrics import safe_auprc

        v_auprc = safe_auprc(bundle.labels[valid_e], 1 / (1 + np.exp(-v_logits)))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "valid_auprc": v_auprc,
        }
        history.append(row)
        if log_epoch_fn is not None:
            log_epoch_fn(row, step=epoch)
        if np.isfinite(v_auprc) and v_auprc > best_valid:
            best_valid = float(v_auprc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibrate on validation
    v_logits = predict_cohort_logits(
        model, bundle, valid_e, mask, cfg.kappa, device, patient_split="valid"
    )
    calibrator = fit_temperature_bias(v_logits, bundle.labels[valid_e])
    return model, calibrator, {"best_valid_auprc": best_valid, "history": history}


def evaluate_split(
    model: FinalEdgeDecoder,
    calibrator: Calibrator,
    bundle: EdgeFeatureBundle,
    mask: VariantFeatureMask,
    kappa: float,
    split: str,
    device: torch.device,
) -> dict:
    idx = _split_index(bundle, split)
    # Use train patients for cohort evidence at test time (no test-label leakage);
    # path/gate features for those patients were built without edge labels.
    logits = predict_cohort_logits(
        model, bundle, idx, mask, kappa, device, patient_split="train"
    )
    probs = calibrator.predict_proba(logits)
    labels = bundle.labels[idx]
    roles = [bundle.negative_roles[i] for i in idx]
    metrics = edge_classification_metrics(labels, probs)
    metrics.update(hits_at_k(labels, probs))
    role_df = role_specific_fpr(labels, probs, roles)
    metrics["role_fpr"] = role_df
    metrics["logits"] = logits
    metrics["probs"] = probs
    metrics["labels"] = labels
    metrics["index"] = idx
    return metrics
