from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def _load_p_linear() -> np.ndarray:
    """Reuse the constrained solve from tsresp to fetch p_linear."""
    from .tsresp import (
        _project_root,
        build_expansion_matrix,
        build_atom_constraint_system,
        build_total_constraint_mask,
        load_bucket_constraints,
        load_symmetry_buckets,
        load_total_constraint,
        load_atom_labels_from_pdb,
        solve_least_squares_with_constraints,
    )
    from protocharge.linearESPcharges.linear import prepare_linear_system

    project_root = _project_root()
    symmetry_buckets = load_symmetry_buckets(
        project_root
        / "data"
        / "microstates"
        / "PPP"
        / "symmetry-buckets"
        / "r8.dat"
    )
    P = build_expansion_matrix(symmetry_buckets)
    atom_count = P.shape[0]
    atom_labels = load_atom_labels_from_pdb(
        project_root / "data" / "microstates" / "PPP" / "ppp.pdb"
    )
    design_matrix, esp_values, _, _ = prepare_linear_system(
        project_root / "data" / "raw" / "resp.out",
        project_root / "data" / "raw" / "esp.xyz",
        atom_count,
    )

    total_charge_target, total_labels = load_total_constraint(
        project_root
        / "data"
        / "microstates"
        / "PPP"
        / "charge-contraints"
        / "total_constraint.yaml"
    )
    total_mask = build_total_constraint_mask(atom_labels, total_labels)
    bucket_constraints = load_bucket_constraints(
        project_root
        / "data"
        / "microstates"
        / "PPP"
        / "charge-contraints"
        / "bucket_constraints.yaml"
    )
    C, d = build_atom_constraint_system(P, total_charge_target, bucket_constraints, total_mask)
    theta, _ = solve_least_squares_with_constraints(
        design_matrix, esp_values, P, C, d
    )
    return P @ theta


def load_mol2(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def find_atom_section(lines: List[str]) -> slice:
    start = end = None
    for idx, line in enumerate(lines):
        if line.startswith("@<TRIPOS>ATOM"):
            start = idx + 1
        elif line.startswith("@<TRIPOS>") and start is not None:
            end = idx
            break
    if start is None:
        raise ValueError("ATOM section not found in mol2 file.")
    if end is None:
        end = len(lines)
    return slice(start, end)


def swap_charges(mol2_lines: List[str], new_charges: np.ndarray) -> List[str]:
    atom_slice = find_atom_section(mol2_lines)
    atom_lines = mol2_lines[atom_slice]
    if len(atom_lines) != new_charges.size:
        raise ValueError(
            f"Charge vector length {new_charges.size} does not match "
            f"mol2 atom count {len(atom_lines)}."
        )

    updated = mol2_lines.copy()
    for idx, (line, charge) in enumerate(zip(atom_lines, new_charges.flat)):
        parts = line.rstrip("\n").split()
        if len(parts) < 9:
            raise ValueError(f"Unexpected ATOM line format: {line!r}")
        parts[8] = f"{charge: .6f}"
        updated[atom_slice.start + idx] = " ".join(parts) + "\n"
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap MOL2 charges with RESP output.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/microstates/PPP/dou.mol2"),
        help="Input MOL2 file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/microstates/PPP/dou_new.mol2"),
        help="Output MOL2 file with swapped charges.",
    )
    args = parser.parse_args()

    mol2_lines = load_mol2(args.input)
    charges = _load_p_linear()
    updated_lines = swap_charges(mol2_lines, charges)
    args.output.write_text("".join(updated_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
