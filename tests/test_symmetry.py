from __future__ import annotations

from itertools import chain
from pathlib import Path

import pytest
from rdkit import Chem

from biliresp.symmetry import buckets_from_graph, buckets_from_pdb, mol_to_nx


def test_mol_to_nx_builds_expected_graph():
    mol = Chem.MolFromSmiles("CC")  # ethane, two equivalent carbon atoms
    graph = mol_to_nx(mol)

    assert graph.number_of_nodes() == mol.GetNumAtoms()
    assert graph.number_of_edges() == mol.GetNumBonds()

    for idx in graph.nodes:
        assert graph.nodes[idx]["element"] == "C"
        assert graph.nodes[idx]["Z"] == 6

    for (u, v, data) in graph.edges(data=True):
        assert data["order"] == pytest.approx(1.0)
        assert not data["aromatic"]

    buckets = buckets_from_graph(graph, radius=2)
    assert buckets == [[0, 1]]


def test_buckets_from_pdb_partitions_atoms():
    repo_root = Path(__file__).resolve().parents[1]
    pdb_path = repo_root / "data" / "raw" / "1.pose.pdb"

    buckets = buckets_from_pdb(pdb_path, radius=2)
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True)

    total_atoms = mol.GetNumAtoms()
    flattened = list(chain.from_iterable(buckets))

    assert len(flattened) == total_atoms
    assert sorted(flattened) == list(range(total_atoms))
    assert len(buckets) > 1
