# Now we want multi-molecule multi-conformer RESP
# This is a bit more complex, but we can still use the same basic approach
# We will use the same RESP class, but we will need to modify it to handle multiple molecules and configurations for each molecule.

# Let me walk through a simple example of how we can do this.
# Imagine the two protonation microstates of PRN. PRN_prot and PRN_deprot.
# We want to fit the RESP charges for both of these microstates simultaneously.
# This is actually a generalization of the multiconfigurational RESP fitting that we have already done.
# We will stack the multiple configuration matricies of each molecule on top of each other 
# and then stuck the charge vector on top of each other as well.
# Example of a stucked vector from PRN_prot and PRN_deprot:
# [q_CA_prot, q_CB_prot, q_CG_prot, q_O1_prot, q_O2_prot, q_HA1_prot, 
# q_HA2_prot, q_HA3_prot, q_HB1_prot, q_HB2_prot, q_H11_prot,
# q_CA_deprot, q_CB_deprot, q_CG_deprot, q_O1_deprot, q_O2_deprot, q_HA1_deprot, 
# q_HA2_deprot, q_HA3_deprot, q_HB1_deprot, q_HB2_deprot]
# It has dimensions of (N+M) x 1, where N is the number of atoms in the first molecule (11) and M 
# is the number of atoms in the second molecule (10), so in total 21x1 in this case.
# Of course we will not neccessarily have only two molecules, in fact later we will have more.
# The advantage of doing this is that now we can fit charges from different molecules simultaneously.
# We will need the following:
# 1. A way to read symmetry buckets just like we did before. For example [q_CA_prot, q_CA_deprot] will be in the same bucket, meaning that their
# charges will be replaced by the same q_CA. 
# 2. Lagrange multipliers to constraint subgroups of charge elements to be equal to some target value. For example,
# Sum(q_CA_prot, q_CB_prot, q_CG_prot, q_O1_prot, q_O2_prot, q_HA1_prot, 
# q_HA2_prot, q_HA3_prot, q_HB1_prot, q_HB2_prot, q_H11_prot) = 0.0
# 3. Identical optimization logic to multiconfigurational (reduced_basic) RESP, meaning that we will
# need two step fitting etc. 

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml

from protocharge.multiconfresp.mcresp import _project_root
from protocharge.paths import microstate_config_root, microstate_output_root, output_root
from protocharge.twostepresp_masked_total.tsresp import (
    build_total_constraint_mask,
    load_total_constraint,
    load_atom_labels_from_pdb,
    load_mask_from_yaml,
    resp_step,
    solve_least_squares_with_constraints,
)


@dataclass
class MoleculeSpec:
    name: str
    root: Path
    coulomb_path: Path
    esp_path: Path
    pdb_path: Path
    mask1_path: Path
    mask2_path: Path
    total_constraint_path: Path
    group_constraint_path: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-molecule, multi-configuration reduced_basic RESP with subgroup total-charge constraints.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="YAML manifest describing molecules and global buckets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ (default: manifest dir / mmresp/charges.npz).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=400,
        help="Newton-Krylov iterations per RESP step.",
    )
    parser.add_argument(
        "--f-tol",
        type=float,
        default=1e-12,
        help="Residual tolerance for RESP steps.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Optional ridge term added to the linear KKT solve (default: 0.0).",
    )
    return parser.parse_args(argv)


def load_manifest(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a YAML mapping.")
    if "molecules" not in data:
        raise ValueError("Manifest must contain a 'molecules' list.")
    return data


def build_molecule_specs(manifest: Dict, base_dir: Path) -> List[MoleculeSpec]:
    specs: List[MoleculeSpec] = []
    for entry in manifest.get("molecules", []):
        name = entry.get("name")
        root_val = entry.get("root")
        if root_val:
            root_path = Path(root_val)
            root = root_path if root_path.is_absolute() else (base_dir / root_path)
        else:
            root = _project_root() / "input" / "microstates" / name
        root = root.resolve()
        output_root = microstate_output_root(name).resolve()
        config_root = microstate_config_root(name).resolve()
        if name is None or not root:
            raise ValueError("Each molecule entry must have a name (and optionally root).")

        def _default(path_key: str, default_rel: Path, *, base: Path | None = None) -> Path:
            val = entry.get(path_key)
            if val:
                candidate = Path(val)
                return candidate if candidate.is_absolute() else ((base or root) / candidate)
            return (base or root) / default_rel

        coulomb_path = _default(
            "coulomb", Path("multiconfRESP/coulomb_matrix.npz"), base=output_root
        )
        esp_path = _default(
            "esp", Path("multiconfRESP/esp_vector.npz"), base=output_root
        )
        pdb_path = _default("pdb", Path(f"{name}.pdb"))
        mask1_path = _default(
            "mask_step1", Path("charge-contraints/mask_step_1.yaml"), base=config_root
        )
        mask2_path = _default(
            "mask_step2", Path("charge-contraints/mask_step_2.yaml"), base=config_root
        )
        total_constraint_path = _default(
            "total_constraint", Path("charge-contraints/total_constraint.yaml"), base=config_root
        )
        group_constraint_val = entry.get("group_constraint")
        group_constraint_path = None
        if group_constraint_val:
            gc_candidate = Path(group_constraint_val)
            group_constraint_path = (
                gc_candidate if gc_candidate.is_absolute() else (config_root / gc_candidate)
            )

        specs.append(
            MoleculeSpec(
                name=name,
                root=root,
                coulomb_path=coulomb_path,
                esp_path=esp_path,
                pdb_path=pdb_path,
                mask1_path=mask1_path,
                mask2_path=mask2_path,
                total_constraint_path=total_constraint_path,
                group_constraint_path=group_constraint_path,
            )
        )
    if not specs:
        raise ValueError("No molecules specified in manifest.")
    return specs


def load_matrix(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"File {path} missing dataset '{key}'.")
        return np.asarray(data[key], dtype=float)


def load_masks(spec: MoleculeSpec, atom_labels: Sequence[str], symmetry_buckets: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    mask1 = load_mask_from_yaml(spec.mask1_path, atom_labels, symmetry_buckets)
    mask2 = load_mask_from_yaml(spec.mask2_path, atom_labels, symmetry_buckets)
    return mask1, mask2


def load_global_buckets(
    path: Path,
    atom_maps: Dict[str, Dict[str, List[int]]],
    offsets: Dict[str, int],
    atom_counts: Dict[str, int],
) -> Tuple[List[List[int]], List[float | None], List[float | None]]:
    """Load global buckets YAML: list of buckets, each bucket is list of strings 'mol:ATOM'.
    Supports optional 'target' per bucket when entry is a mapping with keys
    'bucket'/'atoms' and optional 'target'. Supports numeric indices as atom references.
    Buckets may also specify a 'freeze' value to fix that bucket a priori.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Global buckets file {path} not found.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError("Global buckets must be a YAML list.")
    buckets: List[List[int]] = []
    bucket_targets: List[float | None] = []
    bucket_freeze: List[float | None] = []
    for bucket in raw:
        target = None
        freeze_val = None
        entries: List[str] = []
        if isinstance(bucket, dict):
            entries = bucket.get("bucket") or bucket.get("atoms") or []
            target = bucket.get("target")
            freeze_val = bucket.get("freeze")
            if entries is None:
                entries = []
        elif isinstance(bucket, list):
            entries = bucket
        else:
            raise ValueError("Each bucket must be a list of 'mol:ATOM' or a mapping with bucket/atoms.")
        indices: List[int] = []
        for entry in entries:
            if not isinstance(entry, str) or ":" not in entry:
                raise ValueError(f"Bucket entry {entry!r} must be 'mol:ATOM' or 'mol:INDEX'.")
            mol, label = entry.split(":", 1)
            if mol not in atom_maps:
                raise ValueError(f"Unknown molecule '{mol}' in bucket entry {entry!r}.")
            offset = offsets[mol]
            # allow numeric indices or labels
            if label.isdigit():
                idx = int(label)
                if idx < 0 or idx >= atom_counts[mol]:
                    raise IndexError(f"Index {idx} out of range for molecule {mol}.")
                indices.append(offset + idx)
            else:
                atom_idxs = atom_maps[mol].get(label)
                if atom_idxs is None:
                    raise ValueError(f"Label '{label}' not found in molecule '{mol}'.")
                indices.extend([offset + idx for idx in atom_idxs])
        if not indices:
            raise ValueError("Encountered empty bucket after parsing entries.")
        buckets.append(indices)
        bucket_targets.append(float(target) if target is not None else None)
        bucket_freeze.append(float(freeze_val) if freeze_val is not None else None)
    return buckets, bucket_targets, bucket_freeze


def build_expansion_matrix(buckets: List[List[int]], total_atoms: int) -> np.ndarray:
    if not buckets:
        return np.eye(total_atoms, dtype=float)
    P = np.zeros((total_atoms, len(buckets)), dtype=float)
    for j, bucket in enumerate(buckets):
        for idx in bucket:
            if idx < 0 or idx >= total_atoms:
                raise IndexError(f"Bucket index {idx} out of range for {total_atoms} atoms.")
            if P[idx].any():
                raise ValueError(f"Atom {idx} assigned to multiple buckets in global buckets.")
            P[idx, j] = 1.0
    return P


def _serialize_logger(logger: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    def _convert(value: object) -> object:
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value

    return [{key: _convert(value) for key, value in entry.items()} for entry in logger]


def load_group_constraints(path: Path, atom_count: int) -> Tuple[List[np.ndarray], List[float]]:
    """Load group constraints (indices + target charge) from YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Group constraint file {path} not found.")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if isinstance(data, dict) and "group_constraints" in data:
        groups_raw = data["group_constraints"]
    else:
        groups_raw = data

    if not isinstance(groups_raw, list):
        raise ValueError("Group constraint file must be a YAML list.")

    masks: List[np.ndarray] = []
    targets: List[float] = []
    for entry in groups_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"Each group constraint must be a mapping; got {entry!r}")
        if "group_charge" not in entry:
            raise ValueError("Each group must define group_charge.")
        indices = entry.get("constraint_indices")
        if not isinstance(indices, list):
            raise ValueError("Each group must provide constraint_indices as a list.")
        mask = np.zeros(atom_count, dtype=float)
        for idx in indices:
            if idx < 0 or idx >= atom_count:
                raise IndexError(f"Constraint index {idx} out of range for {atom_count} atoms.")
            mask[idx] = 1.0
        if mask.sum() == 0:
            raise ValueError("Group constraint indices did not select any atoms.")
        masks.append(mask)
        targets.append(float(entry["group_charge"]))
    if not masks:
        raise ValueError("No group constraints found.")
    return masks, targets


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    base_dir = manifest_path.parent
    specs = build_molecule_specs(manifest, base_dir)

    # Load per-molecule data.
    design_blocks: List[np.ndarray] = []
    esp_blocks: List[np.ndarray] = []
    atom_labels_all: List[str] = []
    mask1_list: List[np.ndarray] = []
    mask2_list: List[np.ndarray] = []
    constraint_rows: List[np.ndarray] = []
    constraint_targets: List[float] = []
    offsets: Dict[str, int] = {}
    atom_maps: Dict[str, Dict[str, List[int]]] = {}
    atom_counts: Dict[str, int] = {}

    atom_offset = 0
    for spec in specs:
        offsets[spec.name] = atom_offset
        print(f"Loading Coulomb from {spec.coulomb_path}")
        A = load_matrix(spec.coulomb_path, "A")
        print(f"Loading ESP from {spec.esp_path}")
        Y = load_matrix(spec.esp_path, "Y").reshape(-1)
        if A.shape[0] != Y.size:
            raise ValueError(f"Coulomb/ESP size mismatch for {spec.name}: {A.shape} vs {Y.size}")
        design_blocks.append(A)
        esp_blocks.append(Y)

        labels = load_atom_labels_from_pdb(spec.pdb_path)
        atom_labels_all.extend(labels)
        # map label -> indices (handle duplicates)
        label_to_indices: Dict[str, List[int]] = {}
        for idx_local, lab in enumerate(labels):
            label_to_indices.setdefault(lab, []).append(idx_local)
        atom_maps[spec.name] = label_to_indices
        atom_counts[spec.name] = len(labels)

        if spec.group_constraint_path and spec.group_constraint_path.is_file():
            group_masks, group_targets = load_group_constraints(spec.group_constraint_path, len(labels))
            for mask_vec, target in zip(group_masks, group_targets):
                row = np.zeros(atom_offset + len(labels), dtype=float)
                row[atom_offset : atom_offset + len(labels)] = mask_vec.flatten()
                constraint_rows.append(row)
                constraint_targets.append(target)
        else:
            from protocharge.twostepresp_masked_total.tsresp import load_total_constraint  # local import to avoid cycle issues

            total_charge, total_constraint_labels = load_total_constraint(spec.total_constraint_path)
            total_mask_local = build_total_constraint_mask(labels, total_constraint_labels)
            mask_vec = total_mask_local.flatten() if total_mask_local is not None else np.ones(len(labels), dtype=float)
            row = np.zeros(atom_offset + len(labels), dtype=float)
            row[atom_offset : atom_offset + len(labels)] = mask_vec
            constraint_rows.append(row)
            constraint_targets.append(total_charge)

        # load masks
        # local dummy symmetry buckets per molecule: each atom its own bucket for mask parsing
        sym_dummy = [[i] for i in range(len(labels))]
        m1, m2 = load_masks(spec, labels, sym_dummy)
        # expand to global length later after stacking
        mask1_global = np.zeros(atom_offset + len(labels), dtype=float)
        mask1_global[atom_offset : atom_offset + len(labels)] = m1.flatten()
        mask1_list.append(mask1_global)

        mask2_global = np.zeros(atom_offset + len(labels), dtype=float)
        mask2_global[atom_offset : atom_offset + len(labels)] = m2.flatten()
        mask2_list.append(mask2_global)

        atom_offset += len(labels)

    total_atoms = atom_offset
    # Build block-diagonal design matrix and stacked ESP vector.
    design_matrix = np.zeros((sum(A.shape[0] for A in design_blocks), total_atoms), dtype=float)
    current_row = 0
    current_col = 0
    for A in design_blocks:
        rows, cols = A.shape
        design_matrix[current_row : current_row + rows, current_col : current_col + cols] = A
        current_row += rows
        current_col += cols
    esp_values = np.concatenate(esp_blocks)

    # Global buckets
    global_buckets_path = manifest.get("global_buckets")
    buckets: List[List[int]] = []
    bucket_targets: List[float | None] = []
    bucket_freeze: List[float | None] = []
    if global_buckets_path:
        gb_path = Path(global_buckets_path)
        if not gb_path.is_absolute():
            gb_path = (base_dir / gb_path).resolve()
        buckets, bucket_targets, bucket_freeze = load_global_buckets(gb_path, atom_maps, offsets, atom_counts)
    P = build_expansion_matrix(buckets, total_atoms)

    # Build masks (stacked to total length)
    mask_step1 = np.zeros(total_atoms, dtype=float)
    mask_step2 = np.zeros(total_atoms, dtype=float)
    for m1 in mask1_list:
        mask_step1[: m1.size] += m1
    for m2 in mask2_list:
        mask_step2[: m2.size] += m2

    # Bucket target constraints (global buckets with target)
    for bucket, target, freeze_val in zip(buckets, bucket_targets, bucket_freeze):
        if target is None or freeze_val is not None:
            continue
        row = np.zeros(total_atoms, dtype=float)
        row[bucket] = 1.0
        constraint_rows.append(row)
        constraint_targets.append(target * len(bucket))

    # Pad constraint rows to total atom length
    constraint_matrix = []
    for row in constraint_rows:
        padded = np.zeros(total_atoms, dtype=float)
        padded[: row.size] = row
        constraint_matrix.append(padded)
    C = np.vstack(constraint_matrix)
    d = np.array(constraint_targets, dtype=float).reshape(-1, 1)

    # Handle frozen buckets (fixed charges)
    bucket_freeze_array = np.array([val is not None for val in bucket_freeze], dtype=bool) if bucket_freeze else np.zeros(P.shape[1], dtype=bool)
    P_frozen = P[:, bucket_freeze_array] if bucket_freeze_array.any() else np.zeros((total_atoms, 0))
    theta_frozen = np.array([val for val in bucket_freeze if val is not None], dtype=float).reshape(-1, 1) if bucket_freeze_array.any() else np.zeros((0, 1))
    p_frozen = P_frozen @ theta_frozen if bucket_freeze_array.any() else np.zeros((total_atoms, 1))
    bucket_variable_mask = ~bucket_freeze_array if bucket_freeze else np.ones(P.shape[1], dtype=bool)
    P_var = P[:, bucket_variable_mask]

    design_matrix_var = design_matrix @ P_var
    esp_adjusted = esp_values.reshape(-1, 1) - design_matrix @ p_frozen
    d_adjusted = d - (C @ p_frozen)

    # Linear solve with constraints
    theta_linear, lambda_linear = solve_least_squares_with_constraints(
        design_matrix,
        esp_adjusted,
        P_var,
        C,
        d_adjusted,
        ridge=args.ridge,
    )

    # RESP step one
    logger1, theta_step1, lambda_step1 = resp_step(
        design_matrix_var,
        esp_adjusted.flatten(),
        P_var,
        atom_labels_all,
        C,
        d_adjusted.flatten(),
        mask_step1.reshape(-1, 1),
        a=0.0005,
        b=0.1,
        theta_init=theta_linear,
        lambda_init=lambda_linear,
        maxiter=args.maxiter,
        p_fixed=p_frozen.flatten(),
        description="RESP step one (multi-molecule)",
        print_summary=False,
    )
    charges_step1 = (P_var @ theta_step1 + p_frozen).flatten()

    # RESP step two (variable buckets if mask_step2 selects some atoms)
    theta_final = theta_step1
    logger2: List[Dict[str, float]] = []
    if np.any(mask_step2):
        mask_bool = mask_step2.astype(bool)
        bucket_variable_full = np.array([
            any(mask_bool[idx] for idx in bucket) for bucket in buckets
        ], dtype=bool) if buckets else np.ones(P.shape[1], dtype=bool)
        bucket_variable = bucket_variable_full[bucket_variable_mask]

        if np.any(bucket_variable):
            P_var_step2 = P_var[:, bucket_variable]
            P_fixed_step2 = P_var[:, ~bucket_variable]

            theta_var_init = theta_step1[bucket_variable]
            theta_fixed = theta_step1[~bucket_variable]

            design_var_step2 = design_matrix @ P_var_step2
            esp_col = esp_values.reshape(-1, 1) - design_matrix @ p_frozen
            if np.any(~bucket_variable):
                design_fix = design_matrix @ P_fixed_step2
                esp_adjusted_step2 = esp_col - design_fix @ theta_fixed
                p_fixed = (P_fixed_step2 @ theta_fixed + p_frozen).flatten()
            else:
                esp_adjusted_step2 = esp_col
                p_fixed = p_frozen.flatten()

            esp_adjusted_step2 = esp_adjusted_step2.reshape(-1)
            d_adjusted_step2 = d.flatten() - (C @ p_fixed)

            logger2, theta_var, lambda_step2 = resp_step(
                design_var_step2,
                esp_adjusted_step2,
                P_var_step2,
                atom_labels_all,
                C,
                d_adjusted_step2,
                mask_step2.reshape(-1, 1),
                a=0.001,
                b=0.1,
                theta_init=theta_var_init,
                lambda_init=lambda_step1,
                maxiter=args.maxiter,
                p_fixed=p_fixed,
                description="RESP step two (multi-molecule)",
                print_summary=False,
            )

            theta_full_var = theta_step1.copy()
            theta_full_var[bucket_variable] = theta_var
            theta_final = theta_full_var
    charges_final = (P_var @ theta_final + p_frozen).flatten()

    # Slice outputs per molecule
    charges_step1_blocks: Dict[str, np.ndarray] = {}
    charges_final_blocks: Dict[str, np.ndarray] = {}
    cursor = 0
    for spec in specs:
        labels = load_atom_labels_from_pdb(spec.pdb_path)
        n = len(labels)
        charges_step1_blocks[spec.name] = charges_step1[cursor : cursor + n]
        charges_final_blocks[spec.name] = charges_final[cursor : cursor + n]
        cursor += n

    output_path = args.output or (
        output_root() / args.manifest.stem / "mmresp" / "charges.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        charges_step1=charges_step1,
        charges_final=charges_final,
        charges_step1_by_mol=charges_step1_blocks,
        charges_final_by_mol=charges_final_blocks,
        labels=np.asarray(atom_labels_all, dtype=object),
        logger_step1=_serialize_logger(logger1),
        logger_step2=_serialize_logger(logger2),
        buckets=np.asarray(buckets, dtype=object),
    )
    print(f"Saved multimolecule RESP charges to {output_path}")


if __name__ == "__main__":
    main()
