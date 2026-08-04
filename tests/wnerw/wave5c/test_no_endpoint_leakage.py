from wnerw.wave5c.edge_context_registry import (
    ALLOWED_CONTEXT_TYPES,
    admissible_patient_feature_table,
)


def test_endpoints_and_gates_not_admissible():
    meta = {
        "nsaid": {"node_type": "drug_exposure", "layer": "2_drugs"},
        "ckd": {"node_type": "patient_context", "layer": "1_patient_context"},
        "ddi_renal_double_hit": {
            "node_type": "mechanism",
            "layer": "3_mechanisms",
        },
        "AKI": {"node_type": "clinical_endpoint", "layer": "6_endpoints"},
        "creatinine_rise": {
            "node_type": "adr_or_intermediate_state",
            "layer": "4_intermediate_states",
        },
    }
    df = admissible_patient_feature_table(meta)
    allowed = set(df.loc[df["admissible"], "node"])
    assert "AKI" not in allowed
    assert "ddi_renal_double_hit" not in allowed
    assert "creatinine_rise" not in allowed
    assert "nsaid" in allowed
    assert "ckd" in allowed
    assert ALLOWED_CONTEXT_TYPES == {"drug_exposure", "patient_context"}
