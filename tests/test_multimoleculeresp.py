from pathlib import Path
import numpy as np
import pytest
import yaml

from protocharge.training.multimoleculeresp.mmresp import (
    build_expansion_matrix,
    load_global_buckets,
    load_group_constraints,
)


def test_build_expansion_matrix_unique_assignment():
    buckets = [[0, 2], [1]]
    P = build_expansion_matrix(buckets, total_atoms=3)
    assert P.shape == (3, 2)
    assert np.allclose(P[0], [1, 0])
    assert np.allclose(P[1], [0, 1])
    assert np.allclose(P[2], [1, 0])


def test_load_global_buckets_with_targets_and_freeze(tmp_path: Path):
    data = [
        {"bucket": ["A:CA", "B:0"], "target": 0.5},
        {"atoms": ["A:CB"], "freeze": -0.2},
    ]
    path = tmp_path / "buckets.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    atom_maps = {"A": {"CA": [0], "CB": [1]}, "B": {"0": [0]}}
    offsets = {"A": 0, "B": 2}
    atom_counts = {"A": 2, "B": 1}

    buckets, targets, freezes = load_global_buckets(path, atom_maps, offsets, atom_counts)
    assert buckets == [[0, 2], [1]]
    assert targets == [0.5, None]
    assert freezes == [None, -0.2]


def test_load_group_constraints_indices(tmp_path: Path):
    content = {
        "group_constraints": [
            {"constraint_indices": [0, 2], "group_charge": 1.0},
            {"constraint_indices": [1], "group_charge": -1.0},
        ]
    }
    path = tmp_path / "group.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")

    masks, targets = load_group_constraints(path, atom_count=3)
    assert len(masks) == 2
    assert targets == [1.0, -1.0]
    assert np.allclose(masks[0].flatten(), [1, 0, 1])
    assert np.allclose(masks[1].flatten(), [0, 1, 0])
