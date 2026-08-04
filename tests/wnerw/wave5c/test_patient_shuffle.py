from wnerw.wave5c.runtime import shuffle_patient_profiles


def test_patient_shuffle_is_permutation():
    ids = ["p1", "p2", "p3", "p4"]
    mapping = shuffle_patient_profiles(ids, seed=7)
    assert set(mapping.keys()) == set(ids)
    assert set(mapping.values()) == set(ids)
