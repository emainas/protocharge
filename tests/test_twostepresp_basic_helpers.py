import numpy as np
import pytest

from protocharge.training.twostepresp_basic.tsresp import build_atom_constraint_system, build_expansion_matrix


def test_build_expansion_matrix_basic():
    buckets = [[0, 2], [1]]
    P = build_expansion_matrix(buckets)
    assert P.shape == (3, 2)
    assert np.allclose(P[0], [1, 0])
    assert np.allclose(P[1], [0, 1])
    assert np.allclose(P[2], [1, 0])


def test_build_atom_constraint_system_total_and_bucket():
    buckets = [[0, 2], [1]]
    P = build_expansion_matrix(buckets)
    constraints = [{"bucket": 0, "value": 0.5}]
    C, d = build_atom_constraint_system(P, total_charge=1.0, bucket_constraints=constraints)
    # First row is total charge
    assert C.shape == (2, 3)
    assert np.allclose(C[0], [1, 1, 1])
    # Second row is bucket 0 -> atoms 0 and 2
    assert np.allclose(C[1], [1, 0, 1])
    assert np.allclose(d.flatten(), [1.0, 1.0])
