"""Training entrypoint for Task A structural link prediction.

G-TRAIN (message passing) uses only positive training edges.
CAND-* pairs are scored by the link decoder.
Model selection uses validation only; test is evaluated once at the end.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from data.PreprocessingTaskA.load_hetero_recon_data import load_recon_heterodata
from evaluation.synthetic_evaluator import SynEvaluator
from hcr.attach import fit_and_attach_hcr, hcr_enabled
from hcr.attach_wave3b import fit_and_attach_hcr_wave3b
from models.TaskA.hetero_gnn import HeteroReconGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: DictConfig) -> torch.device:
    requested = str(getattr(cfg.training, "device", "auto")).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("training.device=mps but MPS is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown training.device={requested!r} (auto|cpu|cuda|mps)")


class ReconEvaluator(SynEvaluator):
    """Task A adapter for top-level candidate-edge labels."""

    def __init__(self, cfg, node_to_idx=None):
        super().__init__(cfg, node_to_idx=node_to_idx)

    @torch.no_grad()
    def _forward_probs(self, model, data, criterion, device):
        model.eval()
        data = data.to(device)
        logits = model(data).view(-1)
        labels = data.edge_label.float().to(device).view(-1)
        probs = torch.sigmoid(logits)
        loss = criterion(logits, labels)
        return (
            data,
            labels.detach().cpu().numpy(),
            probs.detach().cpu().numpy(),
            float(loss.item()),
        )


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "parameter_count_total": int(total),
        "parameter_count_trainable": int(trainable),
    }


def build_model(cfg: DictConfig, train_data) -> HeteroReconGNN:
    in_channels = int(train_data[train_data.node_types[0]].x.size(-1))
    hidden_channels = int(
        getattr(cfg.model, "hidden_channels", None)
        or getattr(cfg.model, "hidden_dim", 32)
    )
    return HeteroReconGNN(
        cfg=cfg,
        data=train_data,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
    )


def build_run_name(cfg: DictConfig) -> str:
    if "wandb" in cfg and getattr(cfg.wandb, "run_name", None):
        return str(cfg.wandb.run_name)

    scenario = str(cfg.data.dataset.scenario)
    model_name = str(getattr(cfg.model, "conv_type", cfg.model.name))
    num_layers = int(cfg.model.num_layers)
    training_seed = int(cfg.training.seed)
    feature_ablation_profile = str(
        getattr(cfg.data, "feature_ablation_profile", "empirical")
    ).strip().lower()
    experiment = getattr(cfg, "experiment", None)
    wave = str(getattr(experiment, "wave", "BAZA") if experiment is not None else "BAZA")
    if hcr_enabled(cfg):
        hcr_variant = str(getattr(cfg.hcr, "variant", "unknown"))
    else:
        hcr_variant = str(
            getattr(experiment, "hcr_variant", "none") if experiment is not None else "none"
        )
    return (
        f"{wave}"
        f"__{scenario}"
        f"__{model_name}"
        f"__L{num_layers}"
        f"__seed{training_seed}"
        f"__feat-{feature_ablation_profile}"
        f"__hcr-{hcr_variant}"
    )


def train_epoch(model, data, optimizer, criterion, device: torch.device) -> float:
    model.train()
    data = data.to(device)
    labels = data.edge_label.float().to(device).view(-1)
    optimizer.zero_grad(set_to_none=True)
    logits = model(data).view(-1)
    loss = criterion(logits, labels)
    # Wave 9: optional spline L1 from KAN pair encoders (0 for MLP).
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "pair_encoder_regularization_loss"):
        reg = decoder.pair_encoder_regularization_loss()
        if torch.is_tensor(reg) and float(reg.detach()) != 0.0:
            loss = loss + reg.to(device=loss.device)
    loss.backward()
    clip = getattr(model, "grad_clip", None)
    if clip is not None and float(clip) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
    optimizer.step()
    return float(loss.item())


def prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[f"{prefix}/{key}"] = float(value)
    return out


def flatten_oversmoothing(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if not str(key).startswith("oversmoothing/"):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[f"{prefix}/{key}"] = float(value)
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed = int(cfg.training.seed)
    set_seed(seed)
    device = resolve_device(cfg)
    print(f"\nUsing device: {device}")

    train_data, valid_data, test_data, node_to_idx = load_recon_heterodata(cfg)

    # Wave 3 HCR: fit dependence features on TRAIN patients only, then attach
    # frozen pair vectors to train/valid/test candidates (no val/test refit).
    # Wave 3B: HCR-3 + oracle co-parent Z when motif_completion is enabled.
    # Wave 4C: latent-gate HCR3(A,B,Y) — never reads gate G patient values.
    # Wave 4D: structural context selection + causal-role audit.
    from experiments.motif_completion import latent_gate_enabled, motif_completion_enabled
    from hcr.attach_wave4c import fit_and_attach_hcr_wave4c
    from hcr.attach_wave4d import fit_and_attach_hcr_wave4d, wave4d_enabled

    wave_name = str(getattr(cfg.experiment, "wave", "")).upper()
    variant_name = str(
        getattr(cfg.experiment, "variant", None) or getattr(cfg.hcr, "variant", "")
    ).upper()
    from hcr.wave7.panel_b.attach import fit_and_attach_panel_b, panel_b_enabled
    from hcr.wave7.panel_b_residual.attach import (
        fit_and_attach_panel_b_residual,
        residual_enabled,
    )
    from hcr.wave7.wave7c.attach import fit_and_attach_wave7c, wave7c_enabled

    if (
        wave7c_enabled(cfg)
        or variant_name.startswith("W7C_")
        or variant_name.startswith("W7D_")
        or variant_name.startswith("W9_")
        or variant_name.startswith("W10_")
        or variant_name.startswith("TASKA_")
        or "WAVE7C" in wave_name
        or "WAVE7D" in wave_name
        or "WAVE9" in wave_name
        or "WAVE10" in wave_name
        or "TASKA_MLP_VS_KAN" in wave_name
    ):
        hcr_encoder = fit_and_attach_wave7c(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
    elif residual_enabled(cfg) or variant_name.startswith("W7BR_") or "PANEL_B_RESIDUAL" in wave_name:
        hcr_encoder = fit_and_attach_panel_b_residual(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
    elif panel_b_enabled(cfg) or (
        "W7B_" in variant_name and not variant_name.startswith("W7BR_")
    ) or ("PANEL_B" in wave_name and "RESIDUAL" not in wave_name):
        hcr_encoder = fit_and_attach_panel_b(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
    elif "WAVE7" in wave_name:
        from hcr.wave7.attach import fit_and_attach_wave7

        hcr_encoder = fit_and_attach_wave7(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
    elif "WAVE5D_ARCH_AUDIT" in wave_name or "L2_ARCH" in wave_name:
        from hcr.attach_l2_structural import fit_and_attach_l2_structural
        from link_prediction.wave5d.fingerprints import log_audit_fingerprints
        from data.patient_matrix import resolve_dataset_paths

        repo_root = Path(__file__).resolve().parents[1]
        split_path = resolve_dataset_paths(cfg)["split_path"]
        log_audit_fingerprints(repo_root, patient_split_path=split_path)
        hcr_encoder = fit_and_attach_l2_structural(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
        log_audit_fingerprints(repo_root, patient_split_path=split_path)
    elif wave4d_enabled(cfg):
        hcr_encoder = fit_and_attach_hcr_wave4d(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
    elif latent_gate_enabled(cfg):
        hcr_encoder = fit_and_attach_hcr_wave4c(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
    elif motif_completion_enabled(cfg):
        hcr_encoder = fit_and_attach_hcr_wave3b(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
    else:
        hcr_encoder = fit_and_attach_hcr(
            cfg,
            train_data,
            valid_data,
            test_data,
            device="cpu",
        )
    if hcr_enabled(cfg) and (
        hcr_encoder is not None or getattr(train_data, "hcr_features", None) is not None
    ):
        # Keep decoder.hcr_dim aligned with the attached feature width.
        # Prefer data.hcr_dim (L2 structural is 24 even when base HCR2 encoder is 8).
        fitted_dim = int(
            getattr(train_data, "hcr_dim", 0)
            or (
                getattr(getattr(hcr_encoder, "config", None), "output_dim", 0)
                if hcr_encoder is not None
                else 0
            )
            or getattr(cfg.hcr, "output_dim", 8)
        )
        if hasattr(cfg.model, "decoder") and cfg.model.decoder is not None:
            OmegaConf.set_struct(cfg, False)
            cfg.model.decoder.hcr_dim = fitted_dim
            OmegaConf.set_struct(cfg, True)

    fingerprint = getattr(train_data, "candidate_fingerprint", None)
    candidate_seed = getattr(train_data, "candidate_seed", None)
    feature_ablation_profile = str(
        getattr(cfg.data, "feature_ablation_profile", "empirical")
    ).strip().lower()
    feature_ablation_seed = int(
        getattr(cfg.data, "feature_ablation_seed", 20260722)
    )
    effective_feature_fingerprint = str(
        getattr(train_data, "feature_fingerprint", "missing")
    )
    graph_fp = str(getattr(train_data, "graph_fingerprint", "missing"))
    removed_relations = list(getattr(train_data, "removed_relations", []) or [])
    intervention = str(getattr(cfg.experiment, "intervention", "none"))
    experiment_name = str(getattr(cfg.experiment, "wave", "unknown"))
    print(
        f"\nCANDIDATE FREEZE"
        f"\n  candidate_seed:       {candidate_seed}"
        f"\n  candidate_fingerprint:{fingerprint}"
        f"\n  training.seed:        {seed}"
        f"\n  scenario:             {cfg.data.dataset.scenario}"
        f"\n  node_feature_profile: {getattr(cfg.data, 'node_feature_profile', None)}"
    )
    print(
        "\nEFFECTIVE INPUT FEATURES"
        f"\n  feature_ablation_profile: {feature_ablation_profile}"
        f"\n  feature_ablation_seed:    {feature_ablation_seed}"
        f"\n  feature_fingerprint:      {effective_feature_fingerprint}"
    )
    print(
        "\nGRAPH / EXPERIMENT"
        f"\n  experiment_name:   {experiment_name}"
        f"\n  intervention:      {intervention}"
        f"\n  removed_relations: {removed_relations}"
        f"\n  graph_fingerprint: {graph_fp}"
    )

    model = build_model(cfg, train_data).to(device)
    # Optional clip from training cfg
    model.grad_clip = float(getattr(cfg.training, "grad_clip", 0.0))

    # Materialize lazy Linear(-1, ...) layers before counting parameters.
    with torch.no_grad():
        _ = model(train_data.to(device))

    # Wave 10: freeze HGT + MLP; train only AG KAN residual (+ optional finetune groups).
    # Residual-only / finetune MUST start from a trained MLP checkpoint — freezing a
    # random HGT+MLP leaves ~chance AUPRC (~0.25) because only 2.5k residual params train.
    decoder = getattr(model, "decoder", None)
    w10_train_mode = None
    if decoder is not None and bool(getattr(decoder, "use_ag_kan_residual", False)):
        res_cfg = getattr(decoder, "ag_kan_residual_cfg", None)
        w10_train_mode = str(getattr(res_cfg, "train_mode", "residual_only")).lower()
        init_ckpt = getattr(cfg.training, "init_from_checkpoint", None)
        if init_ckpt is None and res_cfg is not None:
            init_ckpt = getattr(res_cfg, "init_from_checkpoint", None)
        if not init_ckpt:
            raise RuntimeError(
                "Wave 10 AG KAN residual requires training.init_from_checkpoint "
                "(path to R0/A1 MLP best_model.pt). Freezing a random backbone is invalid."
            )
        init_path = Path(str(init_ckpt))
        if not init_path.is_file():
            raise FileNotFoundError(f"init_from_checkpoint not found: {init_path}")
        blob = torch.load(init_path, map_location="cpu", weights_only=False)
        state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
        from models.TaskA.checkpoint_remap import remap_wave7c_mlp_state_dict

        state = remap_wave7c_mlp_state_dict(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Expected missing: ag_kan_residual.* (new branch). Unexpected should be empty.
        missing_non_residual = [k for k in missing if "ag_kan_residual" not in k]
        if missing_non_residual or unexpected:
            raise RuntimeError(
                "Failed to load MLP backbone for Wave 10 residual.\n"
                f"  missing (non-residual): {missing_non_residual[:20]}\n"
                f"  unexpected: {list(unexpected)[:20]}"
            )
        print(
            f"\nWAVE10 INIT FROM CHECKPOINT\n  path: {init_path}\n"
            f"  missing_residual_keys: {len(missing)}\n"
            f"  loaded_backbone: ok"
        )
        for p in model.parameters():
            p.requires_grad = False
        for p in decoder.ag_kan_trainable_parameters():
            p.requires_grad = True
        if w10_train_mode == "finetune":
            for p in decoder.ag_mlp_parameters():
                p.requires_grad = True
            for p in decoder.decoder_head_parameters():
                p.requires_grad = True
            # HGT encoder stays frozen.
        print(
            f"\nWAVE10 FREEZE\n  train_mode: {w10_train_mode}\n"
            f"  trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    param_counts = count_parameters(model)
    print(
        "\nPARAMETER COUNT"
        f"\n  total:     {param_counts['parameter_count_total']:,}"
        f"\n  trainable: {param_counts['parameter_count_trainable']:,}"
    )

    optimizer_name = str(getattr(cfg.training, "optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))
    # Residual fusion: lower LR for legacy encoder + gate (default 3e-4).
    residual_lr = float(getattr(cfg.training, "residual_lr", 3e-4))
    use_param_groups = (
        getattr(model, "residual_fusion", False)
        and decoder is not None
        and hasattr(decoder, "legacy_and_gate_parameters")
        and bool(getattr(decoder, "use_legacy_residual", False))
    )
    opt_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    if w10_train_mode == "finetune" and decoder is not None:
        lr_kan = float(getattr(cfg.training, "lr_kan", lr))
        lr_gate = float(getattr(cfg.training, "lr_gate", lr_kan))
        lr_ag = float(getattr(cfg.training, "lr_existing_ag_mlp", 1e-4))
        lr_dec = float(getattr(cfg.training, "lr_decoder", 1e-4))
        kan_params = list(decoder.ag_kan_residual.kan.parameters())  # type: ignore[union-attr]
        gate_params = [decoder.ag_kan_residual.gate_logit]  # type: ignore[union-attr]
        ag_params = list(decoder.ag_mlp_parameters())
        dec_params = list(decoder.decoder_head_parameters())
        optimizer = opt_cls(
            [
                {"params": kan_params, "lr": lr_kan},
                {"params": gate_params, "lr": lr_gate},
                {"params": ag_params, "lr": lr_ag},
                {"params": dec_params, "lr": lr_dec},
            ],
            weight_decay=weight_decay,
        )
        print(
            f"\nWAVE10 FINETUNE LRs\n  kan={lr_kan} gate={lr_gate} "
            f"ag_mlp={lr_ag} decoder={lr_dec}"
        )
    elif w10_train_mode == "residual_only" and decoder is not None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = opt_cls(trainable, lr=lr, weight_decay=weight_decay)
    elif use_param_groups:
        legacy_params = list(decoder.legacy_and_gate_parameters())
        legacy_ids = {id(p) for p in legacy_params}
        main_params = [p for p in model.parameters() if id(p) not in legacy_ids]
        optimizer = opt_cls(
            [
                {"params": main_params, "lr": lr},
                {"params": legacy_params, "lr": residual_lr},
            ],
            weight_decay=weight_decay,
        )
        print(
            f"\nOPTIMIZER PARAM GROUPS\n  main_lr: {lr}\n  residual_lr: {residual_lr}\n"
            f"  main_params: {sum(p.numel() for p in main_params):,}\n"
            f"  residual_params: {sum(p.numel() for p in legacy_params):,}"
        )
    else:
        optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_labels = train_data.edge_label.float()
    n_pos = float(train_labels.sum().clamp(min=1.0))
    n_neg = float((train_labels.numel() - train_labels.sum()).clamp(min=1.0))
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = torch.nn.BCEWithLogitsLoss()

    evaluator = ReconEvaluator(cfg, node_to_idx=node_to_idx)

    use_wandb = bool(getattr(cfg.wandb, "enabled", True)) if "wandb" in cfg else False
    if use_wandb:
        wandb_tags = list(getattr(cfg.wandb, "tags", []) or [])
        wandb.init(
            project=str(cfg.wandb.project),
            entity=getattr(cfg.wandb, "entity", None),
            name=build_run_name(cfg),
            group=getattr(cfg.wandb, "group", None),
            job_type=getattr(cfg.wandb, "job_type", "training"),
            tags=wandb_tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.summary["candidate_seed"] = candidate_seed
        wandb.summary["candidate_fingerprint"] = fingerprint
        wandb.summary["training_seed"] = seed
        wandb.summary["experiment_name"] = experiment_name
        wandb.summary["intervention"] = intervention
        wandb.summary["removed_relations"] = removed_relations
        wandb.summary["graph_fingerprint"] = graph_fp
        if wandb.run is not None:
            wandb.config.update(
                {
                    "feature_ablation_profile": feature_ablation_profile,
                    "feature_ablation_seed": feature_ablation_seed,
                    "feature_fingerprint": effective_feature_fingerprint,
                    "experiment_name": experiment_name,
                    "intervention": intervention,
                    "removed_relations": removed_relations,
                    "graph_fingerprint": graph_fp,
                },
                allow_val_change=True,
            )
        wandb.summary["feature_ablation_profile"] = feature_ablation_profile
        wandb.summary["feature_ablation_seed"] = feature_ablation_seed
        wandb.summary["feature_fingerprint"] = effective_feature_fingerprint
        wandb.summary["parameter_count_total"] = param_counts["parameter_count_total"]
        wandb.summary["parameter_count_trainable"] = param_counts[
            "parameter_count_trainable"
        ]
        wandb.summary["encoder_name"] = str(
            getattr(cfg.model, "encoder_name", getattr(cfg.model, "name", ""))
        )
        wandb.summary["hidden_channels"] = int(
            getattr(cfg.model, "hidden_channels", getattr(cfg.model, "hidden_dim", -1))
        )
        if hasattr(cfg.model, "heads"):
            wandb.summary["heads"] = int(cfg.model.heads)
        if hasattr(cfg.model, "relation_aggr"):
            wandb.summary["relation_aggr"] = str(cfg.model.relation_aggr)
        wandb.summary["hcr_enabled"] = bool(hcr_enabled(cfg))
        if hcr_enabled(cfg):
            wandb.summary["hcr_variant"] = str(cfg.hcr.variant)
            wandb.summary["hcr_dim"] = int(getattr(train_data, "hcr_dim", 0) or 0)
            wandb.summary["hcr_n_binary_pairs"] = int(
                getattr(train_data, "hcr_n_binary_pairs", 0) or 0
            )
            wandb.summary["num_layers"] = int(cfg.model.num_layers)

    selection_metric = str(getattr(cfg.training, "selection_metric", "auprc")).strip().lower()
    if selection_metric.startswith("valid/"):
        selection_metric = selection_metric[len("valid/") :]

    patience = int(getattr(cfg.training, "early_stopping_patience", 20))
    min_delta = float(getattr(cfg.training, "early_stopping_min_delta", 0.0))
    epochs = int(cfg.training.epochs)

    best_valid = -float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    last_epoch = 0
    diagnostic_threshold = 0.5

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        train_loss = train_epoch(model, train_data, optimizer, criterion, device)

        # TRAIN/VALID only during the loop — never TEST (no peeking for selection).
        train_metrics = evaluator.evaluate(
            model, train_data, eval_criterion, device, threshold=diagnostic_threshold
        )
        valid_metrics = evaluator.evaluate(
            model, valid_data, eval_criterion, device, threshold=diagnostic_threshold
        )

        current_valid = float(valid_metrics.get(selection_metric, float("nan")))
        if np.isfinite(current_valid) and current_valid > best_valid + min_delta:
            best_valid = current_valid
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_dict: dict[str, float] = {
            "epoch": float(epoch),
            "train/optimization_loss": float(train_loss),
            "classification_threshold_diagnostic": diagnostic_threshold,
            "best/valid_metric": float(best_valid),
            "best/epoch": float(best_epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        log_dict.update(prefix_metrics("train", train_metrics))
        log_dict.update(prefix_metrics("valid", valid_metrics))
        log_dict.update(flatten_oversmoothing(valid_metrics, "valid"))
        gate_stats = getattr(getattr(model, "decoder", None), "last_gate_stats", None)
        if isinstance(gate_stats, dict):
            for gk, gv in gate_stats.items():
                if isinstance(gv, (int, float, np.integer, np.floating)):
                    log_dict[str(gk)] = float(gv)

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        print(
            f"Epoch {epoch:03d} | optim {train_loss:.4f} | "
            f"train AUPRC {float(train_metrics.get('auprc', float('nan'))):.4f} | "
            f"valid AUPRC {float(valid_metrics.get('auprc', float('nan'))):.4f} | "
            f"valid AUC {float(valid_metrics.get('auc', float('nan'))):.4f} | "
            f"best {best_epoch:03d}"
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no improvement in valid/{selection_metric} for {patience} epochs."
            )
            if use_wandb:
                wandb.summary["early_stopped"] = True
                wandb.summary["early_stopped_epoch"] = epoch
            break

    if best_epoch < 0:
        raise RuntimeError(
            f"No valid checkpoint selected (metric={selection_metric!r} never finite)."
        )

    model.load_state_dict(best_state)
    model.to(device)

    final_threshold = float(evaluator.select_threshold(model, valid_data, device))
    print(f"\nFinal threshold (valid): {final_threshold:.6f}")

    final_train = evaluator.evaluate(
        model, train_data, eval_criterion, device, threshold=final_threshold
    )
    final_valid = evaluator.evaluate(
        model, valid_data, eval_criterion, device, threshold=final_threshold
    )
    # TEST exactly once after selection.
    final_test = evaluator.evaluate(
        model, test_data, eval_criterion, device, threshold=final_threshold
    )

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "best_epoch": best_epoch,
            "best_valid_metric_name": selection_metric,
            "best_valid_metric": best_valid,
            "classification_threshold": final_threshold,
            "candidate_seed": candidate_seed,
            "candidate_fingerprint": fingerprint,
            "final_train_metrics": final_train,
            "final_valid_metrics": final_valid,
            "final_test_metrics": final_test,
        },
        ckpt_path,
    )

    final_log: dict[str, float] = {
        "final/classification_threshold": final_threshold,
        "final/best_epoch": float(best_epoch),
        "final/best_valid_metric": float(best_valid),
    }
    final_log.update(prefix_metrics("final_train", final_train))
    final_log.update(prefix_metrics("final_valid", final_valid))
    final_log.update(prefix_metrics("final_test", final_test))
    final_log.update(flatten_oversmoothing(final_valid, "final_valid"))
    final_log.update(flatten_oversmoothing(final_test, "final_test"))

    if use_wandb:
        wandb.log(final_log, step=last_epoch + 1)
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_valid_metric_name"] = selection_metric
        wandb.summary["best_valid_metric"] = best_valid
        wandb.summary["best_valid_AUPRC"] = best_valid
        wandb.summary["classification_threshold"] = final_threshold
        wandb.summary["checkpoint_path"] = str(ckpt_path)
        for key, value in final_test.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                wandb.summary[f"test_{key}"] = float(value)
        for key, value in final_valid.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                wandb.summary[f"valid_{key}"] = float(value)
        wandb.finish()

    print("\nFINAL RESULTS")
    print(f"Best epoch: {best_epoch}")
    print(f"Best valid/{selection_metric}: {best_valid:.6f}")
    print(f"Final threshold: {final_threshold:.6f}")
    print(f"Final valid AUPRC: {float(final_valid.get('auprc', float('nan'))):.6f}")
    print(f"Final test AUPRC:  {float(final_test.get('auprc', float('nan'))):.6f}")
    print(f"Final test AUC:    {float(final_test.get('auc', float('nan'))):.6f}")
    print(f"Saved checkpoint:  {ckpt_path}")
    print(f"Hydra run dir:     {output_dir}")


if __name__ == "__main__":
    main()
