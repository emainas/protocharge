"""Symmetry utilities built on Weisfeiler-Lehman (WL) refinement."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple, Union

import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


NodeLabel = Tuple[str, ...]


def mol_to_nx(mol: Mol) -> nx.Graph:
    """Convert an RDKit molecule into a `networkx.Graph` with useful metadata."""

    if mol is None:
        raise ValueError("mol is None (failed to read structure).")

    graph = nx.Graph()
    conf = mol.GetConformer() if mol.GetNumConformers() else None

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pdb_info = atom.GetPDBResidueInfo()

        position = None
        if conf is not None:
            c = conf.GetAtomPosition(idx)
            position = (float(c.x), float(c.y), float(c.z))

        graph.add_node(
            idx,
            element=atom.GetSymbol(),
            Z=atom.GetAtomicNum(),
            atom_name=(pdb_info.GetName().strip() if pdb_info is not None else None),
            pos=position,
        )

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom_symbols = tuple(sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())))

        graph.add_edge(
            i,
            j,
            atom_syms=atom_symbols,
            order=bond.GetBondTypeAsDouble(),
            aromatic=bond.GetIsAromatic(),
            bond_type=str(bond.GetBondType()),
        )

    return graph


def wl_refine(
    graph: nx.Graph,
    r: int = 2,
    *,
    use_edge_labels: bool = True,
    return_all_r: bool = False,
) -> Union[Mapping[int, NodeLabel], Sequence[Mapping[int, NodeLabel]]]:
    """Run WL color refinement for ``r`` rounds and return the node labels."""

    labels: Dict[int, NodeLabel] = {node: ("Z", str(graph.nodes[node]["Z"])) for node in graph.nodes()}
    all_rounds: List[Mapping[int, NodeLabel]] = [labels]

    for _ in range(r):
        new_labels: Dict[int, NodeLabel] = {}
        for node in graph.nodes():
            neigh_multiset = []
            for neighbor in graph.neighbors(node):
                if use_edge_labels and "order" in graph.edges[node, neighbor]:
                    edge = graph.edges[node, neighbor]
                    edge_label = (
                        "ord",
                        str(edge.get("order")),
                        "arom",
                        str(edge.get("aromatic", False)),
                        "bond",
                        edge.get("bond_type", ""),
                    )
                    neigh_multiset.append((edge_label, labels[neighbor]))
                else:
                    neigh_multiset.append(labels[neighbor])
            neigh_multiset.sort()
            new_labels[node] = ("self",) + labels[node] + ("N", tuple(neigh_multiset))
        labels = new_labels
        all_rounds.append(labels)

    return all_rounds if return_all_r else labels


def wl_equivalence_classes(labels: Mapping[int, NodeLabel]) -> List[List[int]]:
    """Group nodes with identical WL labels into equivalence-class buckets."""

    buckets: MutableMapping[NodeLabel, List[int]] = defaultdict(list)
    for node, label in labels.items():
        buckets[label].append(node)
    return sorted((sorted(bucket) for bucket in buckets.values()), key=lambda bucket: bucket[0])


def buckets_from_graph(
    graph: nx.Graph,
    *,
    radius: int = 2,
    use_edge_labels: bool = True,
) -> List[List[int]]:
    """Convenience function returning WL buckets directly from a graph."""

    labels = wl_refine(graph, r=radius, use_edge_labels=use_edge_labels)
    return wl_equivalence_classes(labels)


def buckets_from_pdb(
    pdb_path: Union[str, Path],
    *,
    radius: int = 2,
    remove_hs: bool = True,
    use_edge_labels: bool = True,
) -> List[List[int]]:
    """Load a PDB and return WL symmetry buckets."""

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=remove_hs)
    graph = mol_to_nx(mol)
    return buckets_from_graph(graph, radius=radius, use_edge_labels=use_edge_labels)
