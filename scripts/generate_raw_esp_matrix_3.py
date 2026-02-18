from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from protocharge.paths import ensure_results_dir

from protocharge.training.linearESPcharges import prepare_linear_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate linear ESP charges for all configurations with multiple subgroup total-charge constraints.",
    )
    parser.add_argument(
        "--microstate-root",
        type=Path,
        required=True,
        help="Path to microstate directory (expects terachem/, rawESP/, etc.).",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        help="PDB file providing atom labels; defaults to the single *.pdb under the microstate root.",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        help="Number of atoms (falls back to PDB atom count).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index inside resp.out files (default: last frame).",
    )
    parser.add_argument(
        "--grid-frame",
        type=int,
        default=0,
        help="ESP grid frame index inside esp.xyz files (default: 0).",
    )
    parser.add_argument(
        "--constraint-indices",
        action="append",
        help="Comma-separated zero-based atom indices for one subgroup; repeat flag for multiple groups.",
    )
    parser.add_argument(
        "--constraint-file",
        action="append",
        type=Path,
        help="File listing zero-based atom indices (one per line) for one subgroup; repeat flag for multiple groups.",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=float,
        help="Target total charge for the matching --constraint-labels/group; repeat to match each group.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ path (default: output/<microstate>/rawESP/charges_constraint_multi.npz).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered configuration pairs and constraint groups without fitting.",
    )
    return parser.parse_args()


def load_atom_labels(pdb_path: Path) -> List[str]:
    labels: List[str] = []
    with pdb_path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                labels.append(atom_name)
    if not labels:
        raise ValueError(f"No ATOM/HETATM entries found in {pdb_path}")
    return labels


def build_file_map(directory: Path, suffix: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        name = path.name
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        mapping[stem] = path
    return mapping


def ensure_single_pdb(root: Path) -> Path:
    pdb_files = sorted(root.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB file found under {root}")
    if len(pdb_files) > 1:
        raise FileNotFoundError(f"Multiple PDB files found under {root}; please specify one with --pdb")
    return pdb_files[0]


def ordered_configs(resp_map: Dict[str, Path], esp_map: Dict[str, Path]) -> List[Tuple[str, Path, Path]]:
    common = sorted(set(resp_map) & set(esp_map))
    return [(stem, resp_map[stem], esp_map[stem]) for stem in common]


def _parse_index_string(raw: str) -> List[int]:
    indices: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            indices.append(int(token))
    return indices


def collect_constraint_groups(arg_indices: Sequence[str] | None, files: Sequence[Path] | None) -> List[List[int]]:
    groups: List[List[int]] = []
    if arg_indices:
        for entry in arg_indices:
            group = _parse_index_string(entry)
            if group:
                groups.append(group)
    if files:
        for path in files:
            group: List[int] = []
            with path.open() as handle:
                for line in handle:
                    token = line.strip()
                    if token:
                        group.append(int(token))
            if group:
                groups.append(group)
    return groups


def build_group_mask(n_atoms: int, selected: Iterable[int]) -> np.ndarray:
    selected_list = list(selected)
    if not selected_list:
        raise ValueError("Constraint group cannot be empty.")
    mask = np.zeros(n_atoms, dtype=np.float64)
    for idx in selected_list:
        if idx < 0 or idx >= n_atoms:
            raise IndexError(f"Constraint index {idx} out of range for {n_atoms} atoms.")
        mask[idx] = 1.0
    if mask.sum() == 0:
        raise ValueError(
            "Constraint indices did not match any atoms. "
            f"Requested: {selected_list}; available range: 0..{n_atoms-1}"
        )
    return mask


def solve_least_squares_with_constraints(
    design_matrix: np.ndarray,
    esp_values: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve min ||Aq - V||^2 subject to C q = d via KKT."""
    A = np.asarray(design_matrix, dtype=float)
    V = np.asarray(esp_values, dtype=float)
    C = np.asarray(constraint_matrix, dtype=float)
    d = np.asarray(constraint_targets, dtype=float).reshape(-1, 1)

    H = A.T @ A
    g = A.T @ V
    if g.ndim == 1:
        g = g.reshape(-1, 1)

    zero_block = np.zeros((C.shape[0], C.shape[0]))
    lhs = np.block([[H, C.T], [C, zero_block]])
    rhs = np.vstack([g, d])

    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    q = solution[: A.shape[1]].reshape(-1, 1)
    lam = solution[A.shape[1] :].reshape(-1, 1)
    return q, lam


def main() -> None:
    args = parse_args()

    if not args.constraint_indices and not args.constraint_file:
        raise ValueError("Provide at least one --constraint-indices or --constraint-file group.")
    if not args.target:
        raise ValueError("Provide one --target per constraint group.")

    microstate_root = args.microstate_root.resolve()
    if not microstate_root.is_dir():
        raise NotADirectoryError(f"Microstate root {microstate_root} does not exist or is not a directory.")

    resp_dir = microstate_root / "terachem" / "respout"
    esp_dir = microstate_root / "terachem" / "espxyz"
    if not resp_dir.is_dir():
        raise FileNotFoundError(f"Missing respout directory: {resp_dir}")
    if not esp_dir.is_dir():
        raise FileNotFoundError(f"Missing espxyz directory: {esp_dir}")

    pdb_path = (args.pdb or ensure_single_pdb(microstate_root)).resolve()
    labels = load_atom_labels(pdb_path)

    n_atoms = args.n_atoms if args.n_atoms is not None else len(labels)
    if n_atoms != len(labels):
        raise ValueError(f"Provided n_atoms ({n_atoms}) disagrees with PDB atom count ({len(labels)}).")

    groups = collect_constraint_groups(args.constraint_indices, args.constraint_file)
    if len(groups) != len(args.target):
        raise ValueError(
            f"Mismatch between constraint groups ({len(groups)}) and targets ({len(args.target)}). "
            "Provide one --target per constraint group, in the same order.",
        )

    constraint_masks: List[np.ndarray] = []
    for group in groups:
        mask = build_group_mask(n_atoms, group)
        constraint_masks.append(mask)

    C = np.vstack(constraint_masks)
    d = np.array(args.target, dtype=float).reshape(-1, 1)

    resp_map = build_file_map(resp_dir, ".resp.out")
    esp_map = build_file_map(esp_dir, ".esp.xyz")
    configs = ordered_configs(resp_map, esp_map)
    if not configs:
        raise FileNotFoundError("No matching configuration pairs found between respout and espxyz directories.")

    print(f"Found {len(configs)} configurations under {microstate_root}.")
    print("Constraint groups:")
    for idx, (group, target) in enumerate(zip(groups, args.target), start=1):
        print(f"  [{idx}] target={target:+.4f} indices={group}")

    if args.dry_run:
        for stem, resp_path, esp_path in configs:
            print(f"{stem}: {resp_path.name} | {esp_path.name}")
        return

    charge_columns: List[np.ndarray] = []
    config_labels: List[str] = []
    rmses: List[float] = []
    group_sums: List[np.ndarray] = []

    for idx, (stem, resp_path, esp_path) in enumerate(configs, start=1):
        A, V, _, _ = prepare_linear_system(
            resp_path,
            esp_path,
            n_atoms,
            frame_index=args.frame,
            grid_frame_index=args.grid_frame,
        )

        q, lam = solve_least_squares_with_constraints(A, V, C, d)
        pred = A @ q
        resid = pred - V.reshape(-1, 1)
        rmse = float(np.sqrt(np.mean(resid**2)))
        constraint_values = (C @ q).flatten()

        charge_columns.append(q.flatten())
        config_labels.append(stem)
        rmses.append(rmse)
        group_sums.append(constraint_values)

        if idx % 100 == 0 or idx == len(configs):
            display_sums = ", ".join(f"{val:+.4f}" for val in constraint_values)
            print(f"[{idx}/{len(configs)}] {stem}  RMSE={rmse:.3e}  group sums={display_sums}")

    charges = np.stack(charge_columns, axis=1)
    labels_array = np.asarray(labels, dtype=object)
    config_array = np.asarray(config_labels, dtype=object)
    rmse_array = np.asarray(rmses, dtype=np.float64)
    group_sums_array = np.stack(group_sums, axis=1) if group_sums else np.zeros((len(groups), 0))

    output_path = args.output or (
        ensure_results_dir(microstate_root.name, "rawESP")
        / "charges_constraint_multi.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        charges=charges,
        labels=labels_array,
        configs=config_array,
        rmse=rmse_array,
        constraint_groups=np.asarray(groups, dtype=object),
        constraint_targets=np.asarray(args.target, dtype=np.float64),
        constraint_sums=group_sums_array,
        pdb=str(pdb_path),
    )
    print(f"Saved aggregated charges to {output_path}")


if __name__ == "__main__":
    main()
