from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from protocharge.paths import ensure_results_dir

from protocharge.linearESPcharges import explicit_solution, prepare_linear_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate linear ESP charges for all configurations of a microstate."
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
        "--output",
        type=Path,
        help="Output NPZ path (default: results/<microstate>/rawESP/charges.npz).",
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


def main() -> None:
    args = parse_args()

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

    for idx, (stem, resp_path, esp_path) in enumerate(configs, start=1):
        A, V, Q, _ = prepare_linear_system(
            resp_path,
            esp_path,
            n_atoms,
            frame_index=args.frame,
            grid_frame_index=args.grid_frame,
        )
        result = solver.fit(A, V, Q)
        charge_columns.append(result["q"])
        config_labels.append(stem)
        rmses.append(result["rmse"])
        summed_charges.append(result["sum_q"])
        if idx % 100 == 0 or idx == len(configs):
            print(f"[{idx}/{len(configs)}] {stem}  RMSE={result['rmse']:.3e}  Σq={result['sum_q']:.6f}")

    charges = np.stack(charge_columns, axis=1)
    labels_array = np.asarray(labels, dtype=object)
    config_array = np.asarray(config_labels, dtype=object)
    rmse_array = np.asarray(rmses, dtype=np.float64)
    sum_array = np.asarray(summed_charges, dtype=np.float64)

    output_path = args.output or (
        ensure_results_dir(microstate_root.name, "rawESP") / "charges.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        charges=charges,
        labels=labels_array,
        configs=config_array,
        rmse=rmse_array,
        sum_q=sum_array,
        pdb=str(pdb_path),
    )
    print(f"Saved aggregated charges to {output_path}")


if __name__ == "__main__":
    main()
