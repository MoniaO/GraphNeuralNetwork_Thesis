from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .binary_features import (
    COMPACT_FEATURE_NAMES,
    binary_pair_features,
    result_to_vector,
)
from .registry import feature_names_for_variant
from .variable_spec import VariableSpec, VariableType


CandidatePair = Tuple[str, str]


@dataclass
class HCRPairEncoderConfig:
    smoothing: float = 0.5
    output_dim: int = 8
    unknown_pair_value: float = 0.0
    variant: str = "binary_compact"
    feature_names: Sequence[str] = field(default_factory=lambda: COMPACT_FEATURE_NAMES)
    # Non binary–binary pairs: zeros (wave-1) or hard error.
    unsupported_pair_mode: str = "zeros"
    # Append m_uv ∈ {0,1} so decoder can tell true-near-zero from unsupported.
    append_supported_mask: bool = False
    # Placebo: keep decoder wiring but zero all statistical features.
    force_zero_features: bool = False


class HCRPairEncoder:
    """Fit pair dependence features on training patients only."""

    def __init__(
        self,
        variable_specs: Mapping[str, VariableSpec],
        config: HCRPairEncoderConfig,
    ) -> None:
        self.variable_specs = dict(variable_specs)
        self.config = config
        self._pair_features: Dict[CandidatePair, np.ndarray] = {}
        self._pair_supported: Dict[CandidatePair, bool] = {}
        self._is_fitted = False
        self._n_binary_pairs = 0
        self._n_unsupported_pairs = 0

        names = list(config.feature_names)
        if not names and config.variant:
            names = list(feature_names_for_variant(config.variant))
        feature_dim = len(names)
        expected = feature_dim + (1 if config.append_supported_mask else 0)
        if int(config.output_dim) != expected:
            # Auto-correct output_dim to selected features (+ optional mask).
            config.output_dim = expected
        self._feature_names = tuple(names)
        self.config = config

    @classmethod
    def from_hydra(cls, hcr_cfg, variable_specs: Mapping[str, VariableSpec]) -> "HCRPairEncoder":
        variant = str(getattr(hcr_cfg, "variant", "binary_compact")).strip().lower()
        force_zero = bool(getattr(hcr_cfg, "force_zero_features", False)) or variant in {
            "all_zero",
            "zero",
            "zero_vector",
        }
        if variant in {"all_zero", "zero", "zero_vector"}:
            # Placebo keeps compact dimensionality.
            names = list(feature_names_for_variant("binary_compact"))
            variant = "all_zero"
        else:
            names = list(feature_names_for_variant(variant)) if variant != "none" else []
            features_cfg = getattr(hcr_cfg, "features", None)
            if features_cfg is not None and names:
                selected = []
                for name in names:
                    flag = getattr(features_cfg, name, True)
                    if bool(flag):
                        selected.append(name)
                names = selected

        append_mask = bool(getattr(hcr_cfg, "append_supported_mask", False))
        output_dim = len(names) + (1 if append_mask else 0)
        config = HCRPairEncoderConfig(
            smoothing=float(getattr(hcr_cfg, "smoothing", 0.5)),
            output_dim=output_dim,
            unknown_pair_value=float(getattr(hcr_cfg, "unknown_pair_value", 0.0)),
            variant=variant,
            feature_names=names,
            unsupported_pair_mode=str(
                getattr(hcr_cfg, "unsupported_pair_mode", "zeros")
            ).strip().lower(),
            append_supported_mask=append_mask,
            force_zero_features=force_zero,
        )
        return cls(variable_specs=variable_specs, config=config)

    def fit(
        self,
        train_patient_df: pd.DataFrame,
        candidate_pairs: Sequence[CandidatePair],
    ) -> "HCRPairEncoder":
        """Fit only on the training-patient partition."""
        pair_features: Dict[CandidatePair, np.ndarray] = {}
        pair_supported: Dict[CandidatePair, bool] = {}
        n_binary = 0
        n_unsupported = 0

        unique_pairs = list(dict.fromkeys(candidate_pairs))

        for source_id, target_id in unique_pairs:
            if source_id not in self.variable_specs:
                raise KeyError(f"Missing VariableSpec for source node {source_id!r}.")
            if target_id not in self.variable_specs:
                raise KeyError(f"Missing VariableSpec for target node {target_id!r}.")

            source_spec = self.variable_specs[source_id]
            target_spec = self.variable_specs[target_id]
            key = (source_id, target_id)

            supported = (
                source_spec.variable_type == VariableType.BINARY
                and target_spec.variable_type == VariableType.BINARY
            )

            if supported:
                x = train_patient_df[source_spec.column_name].to_numpy()
                y = train_patient_df[target_spec.column_name].to_numpy()
                result = binary_pair_features(
                    x=x,
                    y=y,
                    smoothing=self.config.smoothing,
                )
                stats = result_to_vector(result, feature_names=self._feature_names)
                n_binary += 1
            else:
                stats = self._encode_unsupported_pair(source_spec, target_spec)
                n_unsupported += 1

            if self.config.force_zero_features:
                stats = np.zeros_like(stats)

            if self.config.append_supported_mask:
                vector = np.concatenate(
                    [stats, np.array([1.0 if supported else 0.0], dtype=np.float32)]
                )
            else:
                vector = stats

            if vector.shape != (self.config.output_dim,):
                raise RuntimeError(
                    f"Invalid feature dimension for {key}: {vector.shape}"
                )
            pair_features[key] = vector
            pair_supported[key] = bool(supported)

        self._pair_features = pair_features
        self._pair_supported = pair_supported
        self._n_binary_pairs = n_binary
        self._n_unsupported_pairs = n_unsupported
        self._is_fitted = True
        return self

    def transform(
        self,
        candidate_pairs: Sequence[CandidatePair],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Transform does not access validation or test patient rows."""
        if not self._is_fitted:
            raise RuntimeError("HCRPairEncoder must be fitted before transform().")

        default = np.full(
            shape=(self.config.output_dim,),
            fill_value=self.config.unknown_pair_value,
            dtype=np.float32,
        )
        if self.config.append_supported_mask:
            default[-1] = 0.0

        matrix = np.stack(
            [self._pair_features.get(pair, default) for pair in candidate_pairs],
            axis=0,
        )
        return torch.as_tensor(matrix, dtype=torch.float32, device=device)

    def supported_mask(
        self,
        candidate_pairs: Sequence[CandidatePair],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("HCRPairEncoder must be fitted before supported_mask().")
        return np.asarray(
            [bool(self._pair_supported.get(pair, False)) for pair in candidate_pairs],
            dtype=bool,
        )

    def _encode_unsupported_pair(
        self,
        source_spec: VariableSpec,
        target_spec: VariableSpec,
    ) -> np.ndarray:
        mode = self.config.unsupported_pair_mode
        if mode == "error":
            raise NotImplementedError(
                "Unsupported variable pair: "
                f"{source_spec.variable_type} -> {target_spec.variable_type}"
            )
        if mode != "zeros":
            raise ValueError(
                f"Unknown unsupported_pair_mode={mode!r} (use zeros|error)."
            )
        return np.full(
            (len(self._feature_names),),
            fill_value=self.config.unknown_pair_value,
            dtype=np.float32,
        )

    @property
    def n_binary_pairs(self) -> int:
        return self._n_binary_pairs

    @property
    def n_unsupported_pairs(self) -> int:
        return self._n_unsupported_pairs
