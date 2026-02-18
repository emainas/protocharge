from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from protocharge.paths import ensure_results_dir

from protocharge.training.linearESPcharges import explicit_solution, prepare_linear_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate linear ESP charges for all configurations of a microstate (constraint mask capable)."
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
        "--constraint-labels",
        type=str,
        help="Comma-separated atom labels to include in the total charge constraint (defaults to all atoms).",
    )
    parser.add_argument(
        "--constraint-file",
        type=Path,
        help="File listing atom labels (one per line) to include in the total charge constraint.",
    )
    parser.add_argument(
        "--total-constraint-file",
        type=Path,
        help="YAML file with a total_charge entry to set the target total charge explicitly.",
    )
    parser.add_argument(
        "--target",
        type=float,
        help="Explicit target total charge for the (masked) constraint; overrides file/RESP-derived value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ path (default: output/<microstate>/rawESP/charges_constraint.npz).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered configuration pairs without fitting.",
    )
    return parser.parse_args()


def load_atom_labels(pdb_path: Path) -> List[str]:
    labels: List[str] = []
    with pdb_path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                labels.append(f"{atom_name}")
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


def parse_constraint_labels(arg_labels: str | None, file_path: Path | None) -> List[str]:
    collected: List[str] = []
    if arg_labels:
        for token in arg_labels.split(","):
            token = token.strip()
            if token:
                collected.append(token)
    if file_path:
        with file_path.open() as handle:
            for line in handle:
                token = line.strip()
                if token:
                    collected.append(token)
    return collected


def build_constraint_mask(all_labels: List[str], selected: Iterable[str]) -> np.ndarray | None:
    selected_list = list(selected)
    if not selected_list:
        return None
    selected_set = set(selected_list)
    mask = np.array([1.0 if label in selected_set else 0.0 for label in all_labels], dtype=np.float64)
    if mask.sum() == 0:
        raise ValueError(
            "Constraint labels did not match any atoms. Requested: "
            f"{', '.join(selected_list)}; available: {', '.join(all_labels)}"
        )
    missing = [name for name in selected_list if name not in all_labels]
    if missing:
        raise ValueError(f"Missing constraint labels: {', '.join(missing)}")
    return mask


def main() -> None:
    args = parse_args()

    if args.target is not None and args.total_constraint_file is not None:
        raise ValueError("Provide only one of --target or --total-constraint-file, not both.")

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

    constraint_names = parse_constraint_labels(args.constraint_labels, args.constraint_file)
    constraint_mask = build_constraint_mask(labels, constraint_names)

    resp_map = build_file_map(resp_dir, ".resp.out")
    esp_map = build_file_map(esp_dir, ".esp.xyz")
    configs = ordered_configs(resp_map, esp_map)
    if not configs:
        raise FileNotFoundError("No matching configuration pairs found between respout and espxyz directories.")

    print(f"Found {len(configs)} configurations under {microstate_root}.")
    if args.dry_run:
        for stem, resp_path, esp_path in configs:
            print(f"{stem}: {resp_path.name} | {esp_path.name}")
        return

    solver = explicit_solution()
    charge_columns: List[np.ndarray] = []
    config_labels: List[str] = []
    rmses: List[float] = []
    summed_charges: List[float] = []
    summed_constraint_charges: List[float] = []

    explicit_target: float | None = None
    if args.target is not None:
        explicit_target = float(args.target)
    elif args.total_constraint_file is not None:
        import yaml

        data = yaml.safe_load(args.total_constraint_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "total_charge" not in data:
            raise ValueError(
                f"Constraint file {args.total_constraint_file} must contain a top-level total_charge entry."
            )
        explicit_target = float(data["total_charge"])

    for idx, (stem, resp_path, esp_path) in enumerate(configs, start=1):
        A, V, _, esp_charges = prepare_linear_system(
            resp_path,
            esp_path,
            n_atoms,
            frame_index=args.frame,
            grid_frame_index=args.grid_frame,
        )

        if explicit_target is not None:
            target_Q = explicit_target
        else:
            target_Q = float(esp_charges @ constraint_mask) if constraint_mask is not None else float(esp_charges.sum())
        result = solver.fit(A, V, target_Q, constraint_mask=constraint_mask)

        charge_columns.append(result["q"])
        config_labels.append(stem)
        rmses.append(result["rmse"])
        summed_charges.append(result["sum_q"])
        summed_constraint_charges.append(result.get("sum_q_constraint", target_Q))

        if idx % 100 == 0 or idx == len(configs):
            print(
                f"[{idx}/{len(configs)}] {stem}  RMSE={result['rmse']:.3e}  "
                f"Σq_all={result['sum_q']:.6f}  Σq_mask={summed_constraint_charges[-1]:.6f}"
            )

    charges = np.stack(charge_columns, axis=1)
    labels_array = np.asarray(labels, dtype=object)
    config_array = np.asarray(config_labels, dtype=object)
    rmse_array = np.asarray(rmses, dtype=np.float64)
    sum_array = np.asarray(summed_charges, dtype=np.float64)
    sum_constraint_array = np.asarray(summed_constraint_charges, dtype=np.float64)

    output_path = args.output or (
        ensure_results_dir(microstate_root.name, "rawESP") / "charges_constraint.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        charges=charges,
        labels=labels_array,
        configs=config_array,
        rmse=rmse_array,
        sum_q=sum_array,
        sum_q_constraint=sum_constraint_array,
        constraint_labels=np.asarray(constraint_names, dtype=object),
        pdb=str(pdb_path),
    )
    print(f"Saved aggregated charges to {output_path}")


if __name__ == "__main__":
    main()
