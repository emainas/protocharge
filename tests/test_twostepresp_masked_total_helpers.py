from pathlib import Path
import numpy as np
import pytest
import yaml

from protocharge.twostepresp_masked_total.tsresp import (
    build_total_constraint_mask,
    load_total_constraint,
    build_atom_constraint_system,
    build_expansion_matrix,
)


def test_load_total_constraint_with_labels(tmp_path: Path):
    data = {"total_charge": 1.0, "constraint_labels": ["CA", "CB"]}
    path = tmp_path / "total.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    total, labels = load_total_constraint(path)
    assert total == 1.0
    assert labels == ["CA", "CB"]


def test_build_total_constraint_mask():
    labels = ["CA", "CB", "CG"]
    mask = build_total_constraint_mask(labels, ["CA", "CG"])
    assert np.allclose(mask.flatten(), [1, 0, 1])


def test_build_atom_constraint_system_with_mask():
    buckets = [[0, 1], [2]]
    P = build_expansion_matrix(buckets)
    constraints = []
    mask = np.array([1.0, 0.0, 1.0]).reshape(-1, 1)
    C, d = build_atom_constraint_system(P, total_charge=0.5, bucket_constraints=constraints, total_charge_mask=mask)
    assert C.shape == (1, 3)
    assert np.allclose(C[0], [1, 0, 1])
    assert np.allclose(d.flatten(), [0.5])
