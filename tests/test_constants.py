from protocharge.constants.atomic_masses import atomic_masses


def test_atomic_masses_has_common_elements():
    assert "H" in atomic_masses
    assert "C" in atomic_masses
    assert atomic_masses["H"] > 0
    assert atomic_masses["C"] > 0
