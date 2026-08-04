"""Frozen Task A / GSN v3 variable-type registry.

Types come from nodes.csv value_type when present; remaining columns
were frozen once from clean-scenario samples with explicit rules
(binary={0,1}, count=integer max<=20, else continuous).
Training must NOT re-infer types from validation/test patients.
"""

from __future__ import annotations

from .variable_spec import VariableSpec, VariableType

VARIABLE_SPECS: dict[str, VariableSpec] = {
    "age": VariableSpec(
        node_id="age",
        variable_type=VariableType.CONTINUOUS,
        column_name="age",
    ),  # source=dataset_freeze_rule
    "alcohol_use": VariableSpec(
        node_id="alcohol_use",
        variable_type=VariableType.BINARY,
        column_name="alcohol_use",
    ),  # source=dataset_freeze_rule
    "baseline_alt_ast": VariableSpec(
        node_id="baseline_alt_ast",
        variable_type=VariableType.CONTINUOUS,
        column_name="baseline_alt_ast",
    ),  # source=dataset_freeze_rule
    "baseline_egfr": VariableSpec(
        node_id="baseline_egfr",
        variable_type=VariableType.CONTINUOUS,
        column_name="baseline_egfr",
    ),  # source=dataset_freeze_rule
    "baseline_mood_vulnerability": VariableSpec(
        node_id="baseline_mood_vulnerability",
        variable_type=VariableType.BINARY,
        column_name="baseline_mood_vulnerability",
    ),  # source=dataset_freeze_rule
    "baseline_potassium": VariableSpec(
        node_id="baseline_potassium",
        variable_type=VariableType.CONTINUOUS,
        column_name="baseline_potassium",
    ),  # source=dataset_freeze_rule
    "baseline_qt_risk": VariableSpec(
        node_id="baseline_qt_risk",
        variable_type=VariableType.BINARY,
        column_name="baseline_qt_risk",
    ),  # source=dataset_freeze_rule
    "baseline_sodium": VariableSpec(
        node_id="baseline_sodium",
        variable_type=VariableType.CONTINUOUS,
        column_name="baseline_sodium",
    ),  # source=dataset_freeze_rule
    "ckd": VariableSpec(
        node_id="ckd",
        variable_type=VariableType.BINARY,
        column_name="ckd",
    ),  # source=dataset_freeze_rule
    "depression_history": VariableSpec(
        node_id="depression_history",
        variable_type=VariableType.BINARY,
        column_name="depression_history",
    ),  # source=dataset_freeze_rule
    "diabetes": VariableSpec(
        node_id="diabetes",
        variable_type=VariableType.BINARY,
        column_name="diabetes",
    ),  # source=dataset_freeze_rule
    "frailty": VariableSpec(
        node_id="frailty",
        variable_type=VariableType.BINARY,
        column_name="frailty",
    ),  # source=dataset_freeze_rule
    "heart_failure": VariableSpec(
        node_id="heart_failure",
        variable_type=VariableType.BINARY,
        column_name="heart_failure",
    ),  # source=dataset_freeze_rule
    "infection": VariableSpec(
        node_id="infection",
        variable_type=VariableType.BINARY,
        column_name="infection",
    ),  # source=dataset_freeze_rule
    "liver_disease": VariableSpec(
        node_id="liver_disease",
        variable_type=VariableType.BINARY,
        column_name="liver_disease",
    ),  # source=dataset_freeze_rule
    "monitoring_intensity": VariableSpec(
        node_id="monitoring_intensity",
        variable_type=VariableType.CONTINUOUS,
        column_name="monitoring_intensity",
    ),  # source=dataset_freeze_rule
    "polypharmacy_burden": VariableSpec(
        node_id="polypharmacy_burden",
        variable_type=VariableType.COUNT,
        column_name="polypharmacy_burden",
    ),  # source=dataset_freeze_rule
    "sex_female": VariableSpec(
        node_id="sex_female",
        variable_type=VariableType.BINARY,
        column_name="sex_female",
    ),  # source=dataset_freeze_rule
    "unobserved_severity": VariableSpec(
        node_id="unobserved_severity",
        variable_type=VariableType.CONTINUOUS,
        column_name="unobserved_severity",
    ),  # source=dataset_freeze_rule
    "acei_arb": VariableSpec(
        node_id="acei_arb",
        variable_type=VariableType.BINARY,
        column_name="acei_arb",
    ),  # source=dataset_freeze_rule
    "aminoglycoside": VariableSpec(
        node_id="aminoglycoside",
        variable_type=VariableType.BINARY,
        column_name="aminoglycoside",
    ),  # source=dataset_freeze_rule
    "antibiotic_hepatic_risk": VariableSpec(
        node_id="antibiotic_hepatic_risk",
        variable_type=VariableType.BINARY,
        column_name="antibiotic_hepatic_risk",
    ),  # source=dataset_freeze_rule
    "anticholinergic": VariableSpec(
        node_id="anticholinergic",
        variable_type=VariableType.BINARY,
        column_name="anticholinergic",
    ),  # source=dataset_freeze_rule
    "anticoagulant": VariableSpec(
        node_id="anticoagulant",
        variable_type=VariableType.BINARY,
        column_name="anticoagulant",
    ),  # source=dataset_freeze_rule
    "antiplatelet": VariableSpec(
        node_id="antiplatelet",
        variable_type=VariableType.BINARY,
        column_name="antiplatelet",
    ),  # source=dataset_freeze_rule
    "benzodiazepine": VariableSpec(
        node_id="benzodiazepine",
        variable_type=VariableType.BINARY,
        column_name="benzodiazepine",
    ),  # source=dataset_freeze_rule
    "beta_blocker": VariableSpec(
        node_id="beta_blocker",
        variable_type=VariableType.BINARY,
        column_name="beta_blocker",
    ),  # source=dataset_freeze_rule
    "contrast_agent": VariableSpec(
        node_id="contrast_agent",
        variable_type=VariableType.BINARY,
        column_name="contrast_agent",
    ),  # source=dataset_freeze_rule
    "corticosteroid": VariableSpec(
        node_id="corticosteroid",
        variable_type=VariableType.BINARY,
        column_name="corticosteroid",
    ),  # source=dataset_freeze_rule
    "diuretic": VariableSpec(
        node_id="diuretic",
        variable_type=VariableType.BINARY,
        column_name="diuretic",
    ),  # source=dataset_freeze_rule
    "hepatotoxic_drug_A": VariableSpec(
        node_id="hepatotoxic_drug_A",
        variable_type=VariableType.BINARY,
        column_name="hepatotoxic_drug_A",
    ),  # source=dataset_freeze_rule
    "herbal_exposure": VariableSpec(
        node_id="herbal_exposure",
        variable_type=VariableType.BINARY,
        column_name="herbal_exposure",
    ),  # source=dataset_freeze_rule
    "nsaid": VariableSpec(
        node_id="nsaid",
        variable_type=VariableType.BINARY,
        column_name="nsaid",
    ),  # source=dataset_freeze_rule
    "opioid": VariableSpec(
        node_id="opioid",
        variable_type=VariableType.BINARY,
        column_name="opioid",
    ),  # source=dataset_freeze_rule
    "ppi": VariableSpec(
        node_id="ppi",
        variable_type=VariableType.BINARY,
        column_name="ppi",
    ),  # source=dataset_freeze_rule
    "qt_prolonging_drug": VariableSpec(
        node_id="qt_prolonging_drug",
        variable_type=VariableType.BINARY,
        column_name="qt_prolonging_drug",
    ),  # source=dataset_freeze_rule
    "snri": VariableSpec(
        node_id="snri",
        variable_type=VariableType.BINARY,
        column_name="snri",
    ),  # source=dataset_freeze_rule
    "ssri": VariableSpec(
        node_id="ssri",
        variable_type=VariableType.BINARY,
        column_name="ssri",
    ),  # source=dataset_freeze_rule
    "thiazide": VariableSpec(
        node_id="thiazide",
        variable_type=VariableType.BINARY,
        column_name="thiazide",
    ),  # source=dataset_freeze_rule
    "valproate_like_drug": VariableSpec(
        node_id="valproate_like_drug",
        variable_type=VariableType.BINARY,
        column_name="valproate_like_drug",
    ),  # source=dataset_freeze_rule
    "anticholinergic_burden": VariableSpec(
        node_id="anticholinergic_burden",
        variable_type=VariableType.BINARY,
        column_name="anticholinergic_burden",
    ),  # source=dataset_freeze_rule
    "cardiac_repolarization_delay": VariableSpec(
        node_id="cardiac_repolarization_delay",
        variable_type=VariableType.BINARY,
        column_name="cardiac_repolarization_delay",
    ),  # source=dataset_freeze_rule
    "cholestatic_pattern": VariableSpec(
        node_id="cholestatic_pattern",
        variable_type=VariableType.BINARY,
        column_name="cholestatic_pattern",
    ),  # source=dataset_freeze_rule
    "cns_sedation": VariableSpec(
        node_id="cns_sedation",
        variable_type=VariableType.BINARY,
        column_name="cns_sedation",
    ),  # source=dataset_freeze_rule
    "coagulation_impairment": VariableSpec(
        node_id="coagulation_impairment",
        variable_type=VariableType.BINARY,
        column_name="coagulation_impairment",
    ),  # source=dataset_freeze_rule
    "electrolyte_disturbance": VariableSpec(
        node_id="electrolyte_disturbance",
        variable_type=VariableType.BINARY,
        column_name="electrolyte_disturbance",
    ),  # source=dataset_freeze_rule
    "fatigue_pathway": VariableSpec(
        node_id="fatigue_pathway",
        variable_type=VariableType.BINARY,
        column_name="fatigue_pathway",
    ),  # source=dataset_freeze_rule
    "gastric_mucosal_injury": VariableSpec(
        node_id="gastric_mucosal_injury",
        variable_type=VariableType.BINARY,
        column_name="gastric_mucosal_injury",
    ),  # source=dataset_freeze_rule
    "gastric_protection": VariableSpec(
        node_id="gastric_protection",
        variable_type=VariableType.BINARY,
        column_name="gastric_protection",
    ),  # source=dataset_freeze_rule
    "hepatic_metabolic_stress": VariableSpec(
        node_id="hepatic_metabolic_stress",
        variable_type=VariableType.BINARY,
        column_name="hepatic_metabolic_stress",
    ),  # source=dataset_freeze_rule
    "hepatocellular_injury": VariableSpec(
        node_id="hepatocellular_injury",
        variable_type=VariableType.BINARY,
        column_name="hepatocellular_injury",
    ),  # source=dataset_freeze_rule
    "monoamine_disruption": VariableSpec(
        node_id="monoamine_disruption",
        variable_type=VariableType.BINARY,
        column_name="monoamine_disruption",
    ),  # source=dataset_freeze_rule
    "nephrotoxic_stress": VariableSpec(
        node_id="nephrotoxic_stress",
        variable_type=VariableType.BINARY,
        column_name="nephrotoxic_stress",
    ),  # source=dataset_freeze_rule
    "oxidative_hepatic_stress": VariableSpec(
        node_id="oxidative_hepatic_stress",
        variable_type=VariableType.BINARY,
        column_name="oxidative_hepatic_stress",
    ),  # source=dataset_freeze_rule
    "platelet_inhibition": VariableSpec(
        node_id="platelet_inhibition",
        variable_type=VariableType.BINARY,
        column_name="platelet_inhibition",
    ),  # source=dataset_freeze_rule
    "potassium_retention": VariableSpec(
        node_id="potassium_retention",
        variable_type=VariableType.BINARY,
        column_name="potassium_retention",
    ),  # source=dataset_freeze_rule
    "psychomotor_impairment": VariableSpec(
        node_id="psychomotor_impairment",
        variable_type=VariableType.BINARY,
        column_name="psychomotor_impairment",
    ),  # source=dataset_freeze_rule
    "reduced_renal_perfusion": VariableSpec(
        node_id="reduced_renal_perfusion",
        variable_type=VariableType.BINARY,
        column_name="reduced_renal_perfusion",
    ),  # source=dataset_freeze_rule
    "serotonergic_shift": VariableSpec(
        node_id="serotonergic_shift",
        variable_type=VariableType.BINARY,
        column_name="serotonergic_shift",
    ),  # source=dataset_freeze_rule
    "sleep_disruption": VariableSpec(
        node_id="sleep_disruption",
        variable_type=VariableType.BINARY,
        column_name="sleep_disruption",
    ),  # source=dataset_freeze_rule
    "sodium_water_imbalance": VariableSpec(
        node_id="sodium_water_imbalance",
        variable_type=VariableType.BINARY,
        column_name="sodium_water_imbalance",
    ),  # source=dataset_freeze_rule
    "tubular_injury": VariableSpec(
        node_id="tubular_injury",
        variable_type=VariableType.BINARY,
        column_name="tubular_injury",
    ),  # source=dataset_freeze_rule
    "volume_depletion": VariableSpec(
        node_id="volume_depletion",
        variable_type=VariableType.BINARY,
        column_name="volume_depletion",
    ),  # source=dataset_freeze_rule
    "alt_ast_rise": VariableSpec(
        node_id="alt_ast_rise",
        variable_type=VariableType.BINARY,
        column_name="alt_ast_rise",
    ),  # source=dataset_freeze_rule
    "anhedonia_like_symptom": VariableSpec(
        node_id="anhedonia_like_symptom",
        variable_type=VariableType.BINARY,
        column_name="anhedonia_like_symptom",
    ),  # source=dataset_freeze_rule
    "bilirubin_rise": VariableSpec(
        node_id="bilirubin_rise",
        variable_type=VariableType.BINARY,
        column_name="bilirubin_rise",
    ),  # source=dataset_freeze_rule
    "cognitive_slowing": VariableSpec(
        node_id="cognitive_slowing",
        variable_type=VariableType.BINARY,
        column_name="cognitive_slowing",
    ),  # source=dataset_freeze_rule
    "confusion_state": VariableSpec(
        node_id="confusion_state",
        variable_type=VariableType.BINARY,
        column_name="confusion_state",
    ),  # source=dataset_freeze_rule
    "creatinine_rise": VariableSpec(
        node_id="creatinine_rise",
        variable_type=VariableType.BINARY,
        column_name="creatinine_rise",
    ),  # source=dataset_freeze_rule
    "dehydration": VariableSpec(
        node_id="dehydration",
        variable_type=VariableType.BINARY,
        column_name="dehydration",
    ),  # source=dataset_freeze_rule
    "dizziness": VariableSpec(
        node_id="dizziness",
        variable_type=VariableType.BINARY,
        column_name="dizziness",
    ),  # source=dataset_freeze_rule
    "fatigue_symptom": VariableSpec(
        node_id="fatigue_symptom",
        variable_type=VariableType.BINARY,
        column_name="fatigue_symptom",
    ),  # source=dataset_freeze_rule
    "gi_irritation": VariableSpec(
        node_id="gi_irritation",
        variable_type=VariableType.BINARY,
        column_name="gi_irritation",
    ),  # source=dataset_freeze_rule
    "hyperkalemia_state": VariableSpec(
        node_id="hyperkalemia_state",
        variable_type=VariableType.BINARY,
        column_name="hyperkalemia_state",
    ),  # source=dataset_freeze_rule
    "hyponatremia_state": VariableSpec(
        node_id="hyponatremia_state",
        variable_type=VariableType.BINARY,
        column_name="hyponatremia_state",
    ),  # source=dataset_freeze_rule
    "hypotension": VariableSpec(
        node_id="hypotension",
        variable_type=VariableType.BINARY,
        column_name="hypotension",
    ),  # source=dataset_freeze_rule
    "jaundice": VariableSpec(
        node_id="jaundice",
        variable_type=VariableType.BINARY,
        column_name="jaundice",
    ),  # source=dataset_freeze_rule
    "mood_lowering": VariableSpec(
        node_id="mood_lowering",
        variable_type=VariableType.BINARY,
        column_name="mood_lowering",
    ),  # source=dataset_freeze_rule
    "nausea": VariableSpec(
        node_id="nausea",
        variable_type=VariableType.BINARY,
        column_name="nausea",
    ),  # source=dataset_freeze_rule
    "occult_bleeding": VariableSpec(
        node_id="occult_bleeding",
        variable_type=VariableType.BINARY,
        column_name="occult_bleeding",
    ),  # source=dataset_freeze_rule
    "overt_bleeding": VariableSpec(
        node_id="overt_bleeding",
        variable_type=VariableType.BINARY,
        column_name="overt_bleeding",
    ),  # source=dataset_freeze_rule
    "qt_prolongation_state": VariableSpec(
        node_id="qt_prolongation_state",
        variable_type=VariableType.BINARY,
        column_name="qt_prolongation_state",
    ),  # source=dataset_freeze_rule
    "sedation_state": VariableSpec(
        node_id="sedation_state",
        variable_type=VariableType.BINARY,
        column_name="sedation_state",
    ),  # source=dataset_freeze_rule
    "sleep_disturbance_symptom": VariableSpec(
        node_id="sleep_disturbance_symptom",
        variable_type=VariableType.BINARY,
        column_name="sleep_disturbance_symptom",
    ),  # source=dataset_freeze_rule
    "adr_reported": VariableSpec(
        node_id="adr_reported",
        variable_type=VariableType.BINARY,
        column_name="adr_reported",
    ),  # source=dataset_freeze_rule
    "creatinine_tested": VariableSpec(
        node_id="creatinine_tested",
        variable_type=VariableType.BINARY,
        column_name="creatinine_tested",
    ),  # source=dataset_freeze_rule
    "ecg_performed": VariableSpec(
        node_id="ecg_performed",
        variable_type=VariableType.BINARY,
        column_name="ecg_performed",
    ),  # source=dataset_freeze_rule
    "electrolyte_abnormality_recorded": VariableSpec(
        node_id="electrolyte_abnormality_recorded",
        variable_type=VariableType.BINARY,
        column_name="electrolyte_abnormality_recorded",
    ),  # source=dataset_freeze_rule
    "electrolytes_tested": VariableSpec(
        node_id="electrolytes_tested",
        variable_type=VariableType.BINARY,
        column_name="electrolytes_tested",
    ),  # source=dataset_freeze_rule
    "elevated_alt_ast_recorded": VariableSpec(
        node_id="elevated_alt_ast_recorded",
        variable_type=VariableType.BINARY,
        column_name="elevated_alt_ast_recorded",
    ),  # source=dataset_freeze_rule
    "elevated_creatinine_recorded": VariableSpec(
        node_id="elevated_creatinine_recorded",
        variable_type=VariableType.BINARY,
        column_name="elevated_creatinine_recorded",
    ),  # source=dataset_freeze_rule
    "fall_reported": VariableSpec(
        node_id="fall_reported",
        variable_type=VariableType.BINARY,
        column_name="fall_reported",
    ),  # source=dataset_freeze_rule
    "hospital_contact": VariableSpec(
        node_id="hospital_contact",
        variable_type=VariableType.BINARY,
        column_name="hospital_contact",
    ),  # source=dataset_freeze_rule
    "lft_tested": VariableSpec(
        node_id="lft_tested",
        variable_type=VariableType.BINARY,
        column_name="lft_tested",
    ),  # source=dataset_freeze_rule
    "mood_screening_done": VariableSpec(
        node_id="mood_screening_done",
        variable_type=VariableType.BINARY,
        column_name="mood_screening_done",
    ),  # source=dataset_freeze_rule
    "qt_recorded": VariableSpec(
        node_id="qt_recorded",
        variable_type=VariableType.BINARY,
        column_name="qt_recorded",
    ),  # source=dataset_freeze_rule
    "AKI": VariableSpec(
        node_id="AKI",
        variable_type=VariableType.BINARY,
        column_name="AKI",
    ),  # source=dataset_freeze_rule
    "DILI": VariableSpec(
        node_id="DILI",
        variable_type=VariableType.BINARY,
        column_name="DILI",
    ),  # source=dataset_freeze_rule
    "Delirium": VariableSpec(
        node_id="Delirium",
        variable_type=VariableType.BINARY,
        column_name="Delirium",
    ),  # source=dataset_freeze_rule
    "Depression": VariableSpec(
        node_id="Depression",
        variable_type=VariableType.BINARY,
        column_name="Depression",
    ),  # source=dataset_freeze_rule
    "Falls": VariableSpec(
        node_id="Falls",
        variable_type=VariableType.BINARY,
        column_name="Falls",
    ),  # source=dataset_freeze_rule
    "GI_bleeding": VariableSpec(
        node_id="GI_bleeding",
        variable_type=VariableType.BINARY,
        column_name="GI_bleeding",
    ),  # source=dataset_freeze_rule
    "Hospitalization": VariableSpec(
        node_id="Hospitalization",
        variable_type=VariableType.BINARY,
        column_name="Hospitalization",
    ),  # source=dataset_freeze_rule
    "Hyperkalemia": VariableSpec(
        node_id="Hyperkalemia",
        variable_type=VariableType.BINARY,
        column_name="Hyperkalemia",
    ),  # source=dataset_freeze_rule
    "Hyponatremia": VariableSpec(
        node_id="Hyponatremia",
        variable_type=VariableType.BINARY,
        column_name="Hyponatremia",
    ),  # source=dataset_freeze_rule
    "QT_arrhythmia": VariableSpec(
        node_id="QT_arrhythmia",
        variable_type=VariableType.BINARY,
        column_name="QT_arrhythmia",
    ),  # source=dataset_freeze_rule
    "active_drug_count": VariableSpec(
        node_id="active_drug_count",
        variable_type=VariableType.COUNT,
        column_name="active_drug_count",
    ),  # source=nodes.csv:value_type
    "ddi_renal_double_hit": VariableSpec(
        node_id="ddi_renal_double_hit",
        variable_type=VariableType.BINARY,
        column_name="ddi_renal_double_hit",
    ),  # source=nodes.csv:value_type
    "ddi_renal_triple_whammy": VariableSpec(
        node_id="ddi_renal_triple_whammy",
        variable_type=VariableType.BINARY,
        column_name="ddi_renal_triple_whammy",
    ),  # source=nodes.csv:value_type
    "ddi_cns_depression_synergy": VariableSpec(
        node_id="ddi_cns_depression_synergy",
        variable_type=VariableType.BINARY,
        column_name="ddi_cns_depression_synergy",
    ),  # source=nodes.csv:value_type
    "ddi_bleeding_dual": VariableSpec(
        node_id="ddi_bleeding_dual",
        variable_type=VariableType.BINARY,
        column_name="ddi_bleeding_dual",
    ),  # source=nodes.csv:value_type
    "ddi_bleeding_triple": VariableSpec(
        node_id="ddi_bleeding_triple",
        variable_type=VariableType.BINARY,
        column_name="ddi_bleeding_triple",
    ),  # source=nodes.csv:value_type
    "ddi_serotonergic_synergy": VariableSpec(
        node_id="ddi_serotonergic_synergy",
        variable_type=VariableType.BINARY,
        column_name="ddi_serotonergic_synergy",
    ),  # source=nodes.csv:value_type
    "ddi_qt_multidrug_load": VariableSpec(
        node_id="ddi_qt_multidrug_load",
        variable_type=VariableType.COUNT,
        column_name="ddi_qt_multidrug_load",
    ),  # source=nodes.csv:value_type
    "ddi_hepatic_triple_hit": VariableSpec(
        node_id="ddi_hepatic_triple_hit",
        variable_type=VariableType.BINARY,
        column_name="ddi_hepatic_triple_hit",
    ),  # source=nodes.csv:value_type
    "cumulative_adr_burden": VariableSpec(
        node_id="cumulative_adr_burden",
        variable_type=VariableType.CONTINUOUS,
        column_name="cumulative_adr_burden",
    ),  # source=nodes.csv:value_type
    "severe_adr_burden": VariableSpec(
        node_id="severe_adr_burden",
        variable_type=VariableType.BINARY,
        column_name="severe_adr_burden",
    ),  # source=nodes.csv:value_type
    "statin": VariableSpec(
        node_id="statin",
        variable_type=VariableType.BINARY,
        column_name="statin",
    ),  # source=nodes.csv:value_type
    "macrolide": VariableSpec(
        node_id="macrolide",
        variable_type=VariableType.BINARY,
        column_name="macrolide",
    ),  # source=nodes.csv:value_type
    "azole_antifungal": VariableSpec(
        node_id="azole_antifungal",
        variable_type=VariableType.BINARY,
        column_name="azole_antifungal",
    ),  # source=nodes.csv:value_type
    "metformin": VariableSpec(
        node_id="metformin",
        variable_type=VariableType.BINARY,
        column_name="metformin",
    ),  # source=nodes.csv:value_type
    "digoxin": VariableSpec(
        node_id="digoxin",
        variable_type=VariableType.BINARY,
        column_name="digoxin",
    ),  # source=nodes.csv:value_type
    "triptan": VariableSpec(
        node_id="triptan",
        variable_type=VariableType.BINARY,
        column_name="triptan",
    ),  # source=nodes.csv:value_type
    "lithium": VariableSpec(
        node_id="lithium",
        variable_type=VariableType.BINARY,
        column_name="lithium",
    ),  # source=nodes.csv:value_type
    "potassium_supplement": VariableSpec(
        node_id="potassium_supplement",
        variable_type=VariableType.BINARY,
        column_name="potassium_supplement",
    ),  # source=nodes.csv:value_type
    "hypertension": VariableSpec(
        node_id="hypertension",
        variable_type=VariableType.BINARY,
        column_name="hypertension",
    ),  # source=nodes.csv:value_type
    "atrial_fibrillation": VariableSpec(
        node_id="atrial_fibrillation",
        variable_type=VariableType.BINARY,
        column_name="atrial_fibrillation",
    ),  # source=nodes.csv:value_type
    "epilepsy": VariableSpec(
        node_id="epilepsy",
        variable_type=VariableType.BINARY,
        column_name="epilepsy",
    ),  # source=nodes.csv:value_type
    "hypothyroidism": VariableSpec(
        node_id="hypothyroidism",
        variable_type=VariableType.BINARY,
        column_name="hypothyroidism",
    ),  # source=nodes.csv:value_type
    "obesity": VariableSpec(
        node_id="obesity",
        variable_type=VariableType.BINARY,
        column_name="obesity",
    ),  # source=nodes.csv:value_type
    "nephrotoxin_load": VariableSpec(
        node_id="nephrotoxin_load",
        variable_type=VariableType.COUNT,
        column_name="nephrotoxin_load",
    ),  # source=nodes.csv:value_type
    "hepatic_drug_load": VariableSpec(
        node_id="hepatic_drug_load",
        variable_type=VariableType.COUNT,
        column_name="hepatic_drug_load",
    ),  # source=nodes.csv:value_type
    "qt_drug_load_v3": VariableSpec(
        node_id="qt_drug_load_v3",
        variable_type=VariableType.COUNT,
        column_name="qt_drug_load_v3",
    ),  # source=nodes.csv:value_type
    "cns_depressant_load": VariableSpec(
        node_id="cns_depressant_load",
        variable_type=VariableType.COUNT,
        column_name="cns_depressant_load",
    ),  # source=nodes.csv:value_type
    "bleeding_risk_load": VariableSpec(
        node_id="bleeding_risk_load",
        variable_type=VariableType.COUNT,
        column_name="bleeding_risk_load",
    ),  # source=nodes.csv:value_type
    "serotonergic_load": VariableSpec(
        node_id="serotonergic_load",
        variable_type=VariableType.COUNT,
        column_name="serotonergic_load",
    ),  # source=nodes.csv:value_type
    "cyp_inhibitor_load": VariableSpec(
        node_id="cyp_inhibitor_load",
        variable_type=VariableType.COUNT,
        column_name="cyp_inhibitor_load",
    ),  # source=nodes.csv:value_type
    "ddi_statin_cyp_inhibitor": VariableSpec(
        node_id="ddi_statin_cyp_inhibitor",
        variable_type=VariableType.BINARY,
        column_name="ddi_statin_cyp_inhibitor",
    ),  # source=nodes.csv:value_type
    "ddi_metformin_renal_risk": VariableSpec(
        node_id="ddi_metformin_renal_risk",
        variable_type=VariableType.BINARY,
        column_name="ddi_metformin_renal_risk",
    ),  # source=nodes.csv:value_type
    "ddi_lithium_renal_risk": VariableSpec(
        node_id="ddi_lithium_renal_risk",
        variable_type=VariableType.BINARY,
        column_name="ddi_lithium_renal_risk",
    ),  # source=nodes.csv:value_type
    "ddi_digoxin_electrolyte_risk": VariableSpec(
        node_id="ddi_digoxin_electrolyte_risk",
        variable_type=VariableType.BINARY,
        column_name="ddi_digoxin_electrolyte_risk",
    ),  # source=nodes.csv:value_type
    "ddi_serotonergic_high_load": VariableSpec(
        node_id="ddi_serotonergic_high_load",
        variable_type=VariableType.BINARY,
        column_name="ddi_serotonergic_high_load",
    ),  # source=nodes.csv:value_type
    "drug_disease_nsaid_ckd": VariableSpec(
        node_id="drug_disease_nsaid_ckd",
        variable_type=VariableType.BINARY,
        column_name="drug_disease_nsaid_ckd",
    ),  # source=nodes.csv:value_type
    "drug_disease_nsaid_heart_failure": VariableSpec(
        node_id="drug_disease_nsaid_heart_failure",
        variable_type=VariableType.BINARY,
        column_name="drug_disease_nsaid_heart_failure",
    ),  # source=nodes.csv:value_type
    "drug_disease_qt_baseline_risk": VariableSpec(
        node_id="drug_disease_qt_baseline_risk",
        variable_type=VariableType.BINARY,
        column_name="drug_disease_qt_baseline_risk",
    ),  # source=nodes.csv:value_type
    "drug_disease_hepatic_liver_disease": VariableSpec(
        node_id="drug_disease_hepatic_liver_disease",
        variable_type=VariableType.BINARY,
        column_name="drug_disease_hepatic_liver_disease",
    ),  # source=nodes.csv:value_type
    "drug_disease_anticoagulant_frailty": VariableSpec(
        node_id="drug_disease_anticoagulant_frailty",
        variable_type=VariableType.BINARY,
        column_name="drug_disease_anticoagulant_frailty",
    ),  # source=nodes.csv:value_type
    "muscle_injury": VariableSpec(
        node_id="muscle_injury",
        variable_type=VariableType.BINARY,
        column_name="muscle_injury",
    ),  # source=nodes.csv:value_type
    "lactate_accumulation": VariableSpec(
        node_id="lactate_accumulation",
        variable_type=VariableType.BINARY,
        column_name="lactate_accumulation",
    ),  # source=nodes.csv:value_type
    "serotonin_toxicity": VariableSpec(
        node_id="serotonin_toxicity",
        variable_type=VariableType.BINARY,
        column_name="serotonin_toxicity",
    ),  # source=nodes.csv:value_type
    "digoxin_toxicity": VariableSpec(
        node_id="digoxin_toxicity",
        variable_type=VariableType.BINARY,
        column_name="digoxin_toxicity",
    ),  # source=nodes.csv:value_type
    "lithium_toxicity": VariableSpec(
        node_id="lithium_toxicity",
        variable_type=VariableType.BINARY,
        column_name="lithium_toxicity",
    ),  # source=nodes.csv:value_type
    "Serotonin_syndrome": VariableSpec(
        node_id="Serotonin_syndrome",
        variable_type=VariableType.BINARY,
        column_name="Serotonin_syndrome",
    ),  # source=nodes.csv:value_type
    "Rhabdomyolysis": VariableSpec(
        node_id="Rhabdomyolysis",
        variable_type=VariableType.BINARY,
        column_name="Rhabdomyolysis",
    ),  # source=nodes.csv:value_type
    "Lactic_acidosis": VariableSpec(
        node_id="Lactic_acidosis",
        variable_type=VariableType.BINARY,
        column_name="Lactic_acidosis",
    ),  # source=nodes.csv:value_type
}

