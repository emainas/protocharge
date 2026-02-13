import numpy as np
import pytest

from biliresp.reduced_basic import reduced as reduced_basic
from biliresp.reduced_masked_total import reduced as reduced_masked_total
from biliresp.reduced_group_constraints import reduced as reduced_group_constraints


def _check_nullspace(mod):
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([1.0, 0.0])
    theta_p, Z, rank = mod._nullspace_components(A, b, tol=1e-12)
    # A @ theta_p should match b in least squares
    assert np.allclose(A @ theta_p, b.reshape(-1, 1))
    assert rank == 1
    assert Z.shape[0] == 2


def test_nullspace_components_all_modules():
    for mod in (reduced_basic, reduced_masked_total, reduced_group_constraints):
        _check_nullspace(mod)


def test_project_initial_affine_space():
    A = np.array([[1.0, 0.0]])
    b = np.array([1.0])
    theta_p, Z, _ = reduced_basic._nullspace_components(A, b, tol=1e-12)
    theta_init = np.array([[2.0], [3.0]])
    projected = reduced_basic._project_initial(theta_init, theta_p, Z)
    # projected should satisfy the constraint A @ theta = b
    assert np.allclose(A @ projected, b.reshape(-1, 1))
