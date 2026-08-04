"""Oracle diagnostic registry for Wave 3B dual/triple gates.

This is NOT a deployable context selector — Z comes from G_true co-parents.
Keep in sync with generator GATE_DEFINITIONS / old_gates + audited edges.
"""

from __future__ import annotations

# Dual AND gates used in observed-gate completion (binary parents only).
DUAL_GATES: dict[str, list[str]] = {
    "ddi_renal_double_hit": ["nsaid", "acei_arb"],
    "ddi_cns_depression_synergy": ["opioid", "benzodiazepine"],
    "ddi_bleeding_dual": ["anticoagulant", "antiplatelet"],
    "ddi_serotonergic_synergy": ["ssri", "snri"],
    "ddi_metformin_renal_risk": ["metformin", "ckd"],
    "ddi_lithium_renal_risk": ["lithium", "ckd"],
    "ddi_digoxin_electrolyte_risk": ["digoxin", "electrolyte_disturbance"],
    "drug_disease_nsaid_ckd": ["nsaid", "ckd"],
    "drug_disease_nsaid_heart_failure": ["nsaid", "heart_failure"],
    "drug_disease_anticoagulant_frailty": ["anticoagulant", "frailty"],
}

# Primary downstream outcomes for latent-gate challenge (Experiment 2).
GATE_OUTCOMES: dict[str, str] = {
    "ddi_renal_double_hit": "reduced_renal_perfusion",
    "ddi_cns_depression_synergy": "cns_sedation",
    "ddi_bleeding_dual": "coagulation_impairment",
    "ddi_serotonergic_synergy": "serotonergic_shift",
    "ddi_metformin_renal_risk": "lactate_accumulation",
    "ddi_lithium_renal_risk": "lithium_toxicity",
    "ddi_digoxin_electrolyte_risk": "digoxin_toxicity",
    "drug_disease_nsaid_ckd": "creatinine_rise",
    "drug_disease_nsaid_heart_failure": "reduced_renal_perfusion",
    "drug_disease_anticoagulant_frailty": "overt_bleeding",
    # non-binary / graded parents — for later latent-gate extension
    "ddi_statin_cyp_inhibitor": "muscle_injury",
}

TRIPLE_GATES: dict[str, list[str]] = {
    "ddi_renal_triple_whammy": ["nsaid", "acei_arb", "diuretic"],
    "ddi_bleeding_triple": ["nsaid", "anticoagulant", "antiplatelet"],
    "ddi_hepatic_triple_hit": [
        "hepatotoxic_drug_A",
        "antibiotic_hepatic_risk",
        "valproate_like_drug",
    ],
}

# Protocol aliases (thesis naming ↔ Hydra hcr=)
VARIANT_ALIASES = {
    # Wave 4B observed-gate (HCR3 with gate column)
    "G0": "none",
    "G1": "binary_compact",
    "G2": "hcr3_full",
    "G3": "hcr3_without_a111",
    "G4": "hcr3_shuffled",
    "G5": "hcr3_random_context",
    # Wave 4C latent-gate (HCR without gate column; Y = outcome)
    "L0": "none",
    "L1": "latent_pairwise_aby",
    "L2": "hcr3_full",
    "L3": "hcr3_without_a111",
    "L4": "hcr3_shuffled",
    "L5": "hcr3_random_context",
    # Wave 4D context-role audit
    "D0": "none",
    "D1": "structural_latent_pairwise",
    "D2": "all_context_top1",
    "D3": "structural_context_shuffled",
    "D4": "matched_random_context",
    "D5": "hcr3_selected_capacity_matched",
}
