from pathlib import Path
import numpy as np
import pytest
import yaml

from biliresp.twostepresp_group_constraints.tsresp import (
    build_group_mask_from_indices,
    load_group_constraints,
    build_atom_constraint_system,
    build_expansion_matrix,
)


def test_build_group_mask_from_indices():
    mask = build_group_mask_from_indices(4, [0, 2])
    assert np.allclose(mask.flatten(), [1, 0, 1, 0])


def test_load_group_constraints(tmp_path: Path):
    data = [
        {"constraint_indices": [0, 1], "group_charge": 1.0},
        {"constraint_indices": [2], "group_charge": -1.0},
    ]
    path = tmp_path / "group.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    masks, targets = load_group_constraints(path, atom_count=3)
    assert targets == [1.0, -1.0]
    assert np.allclose(masks[0].flatten(), [1, 1, 0])
    assert np.allclose(masks[1].flatten(), [0, 0, 1])


def test_build_atom_constraint_system_with_groups():
    buckets = [[0, 1], [2]]
    P = build_expansion_matrix(buckets)
    group_masks = [np.array([1, 0, 1]).reshape(-1, 1)]
    group_targets = [0.5]
    bucket_constraints = [{"bucket": 1, "value": -0.5}]
    C, d = build_atom_constraint_system(P, group_masks, group_targets, bucket_constraints)
    assert C.shape == (2, 3)
    assert np.allclose(C[0], [1, 0, 1])
    assert np.allclose(C[1], [0, 0, 1])
    assert np.allclose(d.flatten(), [0.5, -0.5])
