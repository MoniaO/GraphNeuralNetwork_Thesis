"""HCR-3 pair encoder with an explicit third variable Z per candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .triple_features import (
    binary_triple_features,
    shuffled_z_triple_features,
    triple_result_to_vector,
)


CandidatePair = Tuple[str, str]


@dataclass
class HCR3EncoderConfig:
    output_dim: int = 17
    smoothing: float = 0.5
    include_a111: bool = True
    shuffle_z: bool = False
    shuffle_seed: int = 20260722
    unknown_pair_value: float = 0.0
    variant: str = "hcr3_full"


class HCR3PairEncoder:
    """Fit third-order features on train patients for pairs that have a Z."""

    def __init__(self, config: HCR3EncoderConfig) -> None:
        self.config = config
        self._pair_features: Dict[CandidatePair, np.ndarray] = {}
        self._pair_z: Dict[CandidatePair, str] = {}
        self._is_fitted = False
        self.n_with_z = 0
        self.n_without_z = 0

    def fit(
        self,
        train_patient_df: pd.DataFrame,
        candidate_z: Mapping[CandidatePair, str],
    ) -> "HCR3PairEncoder":
        """Fit with y = candidate target (Wave 3B / oracle-gate)."""
        triples = {
            pair: (pair[0], pair[1], z_name) for pair, z_name in candidate_z.items()
        }
        return self.fit_triples(train_patient_df, triples)

    def fit_triples(
        self,
        train_patient_df: pd.DataFrame,
        candidate_triples: Mapping[CandidatePair, tuple[str, str, str]],
        *,
        forbidden_columns: set[str] | None = None,
    ) -> "HCR3PairEncoder":
        """Fit HCR-3 from explicit (x, y, z) columns per candidate pair.

        Wave 4C latent-gate uses ``(A, Y, B)`` while the candidate edge remains
        ``(A, G)`` — gate column G must not appear in the triple.
        """
        features: Dict[CandidatePair, np.ndarray] = {}
        pair_z: Dict[CandidatePair, str] = {}
        n_with = 0
        n_without = 0
        rng = np.random.default_rng(self.config.shuffle_seed)
        forbidden_extra = set(forbidden_columns or ())

        for pair, (x_name, y_name, z_name) in candidate_triples.items():
            used = {x_name, y_name, z_name}
            # Candidate target is the gate G — must never appear in (A,Y,B).
            gate = str(pair[1])
            if gate in used:
                raise ValueError(
                    f"Latent-gate leakage: gate {gate!r} appears in triple "
                    f"{(x_name, y_name, z_name)} for candidate {pair}"
                )
            if forbidden_extra and used.intersection(forbidden_extra):
                bad = used.intersection(forbidden_extra)
                raise ValueError(
                    f"Latent-gate leakage: triple {used} intersects forbidden "
                    f"{bad} for candidate {pair}"
                )
            if (
                x_name not in train_patient_df.columns
                or y_name not in train_patient_df.columns
                or z_name not in train_patient_df.columns
            ):
                features[pair] = np.full(
                    (self.config.output_dim,),
                    self.config.unknown_pair_value,
                    dtype=np.float32,
                )
                n_without += 1
                continue

            x = train_patient_df[x_name].to_numpy()
            y = train_patient_df[y_name].to_numpy()
            z = train_patient_df[z_name].to_numpy()
            try:
                if self.config.shuffle_z:
                    result = shuffled_z_triple_features(
                        x, y, z, rng=rng, seed=self.config.shuffle_seed
                    )
                else:
                    result = binary_triple_features(
                        x, y, z, smoothing=self.config.smoothing
                    )
                vector = triple_result_to_vector(
                    result, include_a111=self.config.include_a111
                )
            except ValueError:
                vector = np.full(
                    (self.config.output_dim,),
                    self.config.unknown_pair_value,
                    dtype=np.float32,
                )
                n_without += 1
                features[pair] = vector
                continue

            if vector.shape != (self.config.output_dim,):
                raise RuntimeError(f"Bad HCR3 dim for {pair}: {vector.shape}")
            features[pair] = vector
            pair_z[pair] = z_name
            n_with += 1

        self._pair_features = features
        self._pair_z = pair_z
        self.n_with_z = n_with
        self.n_without_z = n_without
        self._is_fitted = True
        return self

    def transform(
        self,
        candidate_pairs: Sequence[CandidatePair],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("HCR3PairEncoder must be fitted before transform().")
        default = np.full(
            (self.config.output_dim,),
            self.config.unknown_pair_value,
            dtype=np.float32,
        )
        matrix = np.stack(
            [self._pair_features.get(pair, default) for pair in candidate_pairs],
            axis=0,
        )
        return torch.as_tensor(matrix, dtype=torch.float32, device=device)
