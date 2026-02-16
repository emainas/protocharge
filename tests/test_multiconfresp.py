import numpy as np
import pytest

from protocharge.multiconfresp.mcresp import ConfigurationSystem, stack_configurations


def _config(stem: str, m: int, n: int, total_charge: float) -> ConfigurationSystem:
    design = np.ones((m, n), dtype=float)
    esp = np.arange(m, dtype=float)
    esp_charges = np.linspace(0.0, 1.0, n)
    positions = np.zeros((n, 3), dtype=float)
    return ConfigurationSystem(
        stem=stem,
        design_matrix=design,
        esp_values=esp,
        total_charge=total_charge,
        esp_charges=esp_charges,
        atom_positions_bohr=positions,
    )


def test_stack_configurations_shapes_and_charge():
    c1 = _config("a", 3, 4, 1.0)
    c2 = _config("b", 5, 4, -1.0)
    system = stack_configurations([c1, c2])

    assert system.design_matrix.shape == (8, 4)
    assert system.esp_values.shape == (8,)
    assert system.esp_charges.shape == (4, 2)
    assert system.atom_positions_bohr.shape == (2, 4, 3)
    assert system.config_order == ["a", "b"]
    assert system.total_charge == pytest.approx(0.0)
    assert system.total_charges.shape == (2,)


def test_stack_configurations_requires_same_atom_count():
    c1 = _config("a", 3, 4, 0.0)
    c2 = _config("b", 3, 5, 0.0)
    with pytest.raises(ValueError):
        stack_configurations([c1, c2])
