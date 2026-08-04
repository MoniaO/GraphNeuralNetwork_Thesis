"""Fixed dimensions / gate slot indices for residual fusion (B2 layout unchanged)."""

from __future__ import annotations

PAIR_B2_DIM = 40
PAIR_V0_DIM = 8
PAIR_CLASSICAL4_DIM = 4
MOTIF_B2_DIM = 120
MOTIF_V0_DIM = 24
MOTIF_CLASSICAL4_DIM = 12  # 3 roles × 4
PAIR_LATENT_DIM = 8
MOTIF_PAIR_LATENT_DIM = 24
TRIPLE_DIM = 6
TRIPLE_LATENT_DIM = 4
MOTIF_LATENT_WITH_TRIPLE = 28  # 24 + 4
N_ROLES = 3

# Gate features from enriched B2 40-d: [p_u, p_v, p11, rarity, support]
GATE_SLOTS = (20, 24, 28, 31, 32)
GATE_FEATURE_NAMES = ("p_u", "p_v", "p11", "rarity", "support")

# classical4 ⊂ binary_compact V0 slots
# compact: p11, cond, RD, logOR, phi, MI, joint_support, uncertainty
CLASSICAL4_V0_SLOTS = (0, 2, 3, 4)
CLASSICAL4_FEATURE_NAMES = (
    "p11",
    "risk_difference",
    "log_odds_ratio",
    "phi",
)

# Back-compat alias used by R1/R2 decoder
MOTIF_LATENT_DIM = MOTIF_PAIR_LATENT_DIM
