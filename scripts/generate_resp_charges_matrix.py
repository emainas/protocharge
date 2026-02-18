from __future__ import annotations

import argparse
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from protocharge.paths import ensure_results_dir

from protocharge.linearESPcharges import explicit_solution, prepare_linear_system
from protocharge.resp.resp import HyperbolicRestraint, fit_resp_charges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate RESP charges for all configurations of a microstate."
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
        help="PDB file providing atom labels and geometry fallback; defaults to the single *.pdb under the microstate root.",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        help="Optional multi-frame XYZ file providing atomic symbols for RESP restraints. "
        "When omitted a temporary XYZ file is generated from the supplied PDB.",
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
        "--total-charge",
        type=float,
        help="Override the total molecular charge enforced during RESP fitting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ path (default: output/<microstate>/onestepRESP/charges.npz).",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=HyperbolicRestraint.a,
        help="Hyperbolic restraint parameter a (default: 0.0005).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=HyperbolicRestraint.b,
        help="Hyperbolic restraint parameter b (default: 0.001).",
    )
    parser.add_argument(
        "--q0",
        type=float,
        default=HyperbolicRestraint.q0,
        help="Reference charge q0 for the restraint (default: 0.0).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum Newton-Krylov iterations per configuration.",
    )
    parser.add_argument(
        "--solver-tol",
        type=float,
        default=1e-11,
        help="Newton-Krylov stopping tolerance on the KKT residual.",
    )
    parser.add_argument(
        "--restrain-heavy-only",
        action="store_true",
        help="Apply RESP restraint only to heavy atoms (default: restrain all atoms).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered configuration pairs without fitting.",
    )
    parser.add_argument(
        "--fail-on-nonconvergence",
        action="store_true",
        help="Abort immediately when the RESP solver fails to converge for a configuration.",
    )
    return parser.parse_args()


def parse_pdb_atoms(pdb_path: Path) -> Tuple[List[str], List[str], List[Tuple[float, float, float]]]:
    labels: List[str] = []
    symbols: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            labels.append(line[12:16].strip())
            symbol = line[76:78].strip()
            if not symbol:
                atom_field = "".join(ch for ch in line[12:16] if ch.isalpha())
                if not atom_field:
                    raise ValueError(f"Cannot infer element symbol from PDB line: {line.rstrip()}")
                symbol = atom_field
            symbol = symbol[0].upper() + symbol[1:].lower()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
            symbols.append(symbol)
    if not labels:
        raise ValueError(f"No ATOM/HETATM entries found in {pdb_path}")
    return labels, symbols, coords


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
        raise FileNotFoundError(
            f"Multiple PDB files found under {root}; please specify one with --pdb"
        )
    return pdb_files[0]


def ordered_configs(resp_map: Dict[str, Path], esp_map: Dict[str, Path]) -> List[Tuple[str, Path, Path]]:
    common = sorted(set(resp_map) & set(esp_map))
    return [(stem, resp_map[stem], esp_map[stem]) for stem in common]


def write_xyz(symbols: Sequence[str], coords: Sequence[Tuple[float, float, float]], path: Path) -> None:
    with path.open("w") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write("Generated from PDB for RESP aggregation\n")
        for symbol, (x, y, z) in zip(symbols, coords):
            handle.write(f"{symbol:<2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")


@contextmanager
def ensure_geometry_xyz(
    provided: Path | None, symbols: Sequence[str], coords: Sequence[Tuple[float, float, float]]
) -> Iterator[Path]:
    if provided is not None:
        resolved = provided.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Geometry file {resolved} does not exist.")
        yield resolved
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        geometry_path = Path(tmpdir) / "generated_geometry.xyz"
        write_xyz(symbols, coords, geometry_path)
        yield geometry_path


def main() -> None:
    args = parse_args()

    microstate_root = args.microstate_root.resolve()
    if not microstate_root.is_dir():
        raise NotADirectoryError(
            f"Microstate root {microstate_root} does not exist or is not a directory."
        )

    resp_dir = microstate_root / "terachem" / "respout"
    esp_dir = microstate_root / "terachem" / "espxyz"
    if not resp_dir.is_dir():
        raise FileNotFoundError(f"Missing respout directory: {resp_dir}")
    if not esp_dir.is_dir():
        raise FileNotFoundError(f"Missing espxyz directory: {esp_dir}")

    pdb_path = (args.pdb or ensure_single_pdb(microstate_root)).resolve()
    labels, symbols, coords = parse_pdb_atoms(pdb_path)

    n_atoms = args.n_atoms if args.n_atoms is not None else len(labels)
    if n_atoms != len(labels):
        raise ValueError(
            f"Provided n_atoms ({n_atoms}) disagrees with PDB atom count ({len(labels)})."
        )

    resp_map = build_file_map(resp_dir, ".resp.out")
    esp_map = build_file_map(esp_dir, ".esp.xyz")
    configs = ordered_configs(resp_map, esp_map)
    if not configs:
        raise FileNotFoundError(
            "No matching configuration pairs found between respout and espxyz directories."
        )

    print(f"Found {len(configs)} configurations under {microstate_root}.")
    if args.dry_run:
        for stem, resp_path, esp_path in configs:
            print(f"{stem}: {resp_path.name} | {esp_path.name}")
        return

    restraint = HyperbolicRestraint(a=args.a, b=args.b, q0=args.q0)
    restrain_all = not args.restrain_heavy_only

    charge_columns: List[np.ndarray] = []
    initial_columns: List[np.ndarray] = []
    config_labels: List[str] = []
    rmses: List[float] = []
    rrmses: List[float] = []
    summed_charges: List[float] = []
    lagrange_multipliers: List[float] = []
    loss_terms: List[float] = []
    ls_terms: List[float] = []
    restraint_terms: List[float] = []
    target_total: List[float] = []
    iterations: List[int] = []
    converged: List[bool] = []
    failed_configs: List[str] = []

    with ensure_geometry_xyz(args.geometry, symbols, coords) as geometry_path:
        geometry_meta = (
            str(args.geometry.resolve())
            if args.geometry is not None
            else "generated_from_pdb"
        )
        for idx, (stem, resp_path, esp_path) in enumerate(configs, start=1):
            try:
                result = fit_resp_charges(
                    resp_path,
                    esp_path,
                    geometry_path,
                    n_atoms,
                    frame_index=args.frame,
                    grid_frame_index=args.grid_frame,
                    restraint=restraint,
                    total_charge=args.total_charge,
                    solver_tol=args.solver_tol,
                    maxiter=args.maxiter,
                    restrain_all_atoms=restrain_all,
                )
                converged.append(True)
            except RuntimeError as exc:
                if args.fail_on_nonconvergence:
                    raise
                print(f"[WARN] {stem}: RESP solver failed to converge ({exc}); storing NaNs.")
                failed_configs.append(stem)
                A, V, total_charge, _ = prepare_linear_system(
                    resp_path,
                    esp_path,
                    n_atoms,
                    frame_index=args.frame,
                    grid_frame_index=args.grid_frame,
                )
                linear_solver = explicit_solution()
                linear_result = linear_solver.fit(A, V, total_charge)
                result = {
                    "charges": np.full(n_atoms, np.nan, dtype=np.float64),
                    "initial_charges": linear_result["q"],
                    "sum_q": np.nan,
                    "lagrange_multiplier": np.nan,
                    "loss": np.nan,
                    "ls_term": np.nan,
                    "restraint": np.nan,
                    "target_total_charge": float(total_charge),
                    "loss_history": [],
                    "rmse": np.nan,
                    "rrms": np.nan,
                }
                converged.append(False)
            charge_columns.append(np.asarray(result["charges"], dtype=np.float64))
            initial_columns.append(np.asarray(result["initial_charges"], dtype=np.float64))
            config_labels.append(stem)
            rmses.append(float(result.get("rmse", np.nan)))
            rrmses.append(float(result.get("rrms", np.nan)))
            summed_charges.append(float(result["sum_q"]))
            lagrange_multipliers.append(float(result["lagrange_multiplier"]))
            loss_terms.append(float(result.get("loss", np.nan)))
            ls_terms.append(float(result.get("ls_term", np.nan)))
            restraint_terms.append(float(result.get("restraint", np.nan)))
            target_total.append(float(result.get("target_total_charge", np.nan)))
            iterations.append(len(result.get("loss_history", [])))
            if idx % 100 == 0 or idx == len(configs):
                print(
                    f"[{idx}/{len(configs)}] {stem}  RMSE={rmses[-1]:.3e}  Σq={summed_charges[-1]:.6f} "
                    f"λ={lagrange_multipliers[-1]:.3e}  converged={converged[-1]}"
                )

    charges = np.stack(charge_columns, axis=1)
    initial = np.stack(initial_columns, axis=1)
    labels_array = np.asarray(labels, dtype=object)
    symbols_array = np.asarray(symbols, dtype=object)
    config_array = np.asarray(config_labels, dtype=object)

    rmse_array = np.asarray(rmses, dtype=np.float64)
    rrms_array = np.asarray(rrmses, dtype=np.float64)
    sum_array = np.asarray(summed_charges, dtype=np.float64)
    lambda_array = np.asarray(lagrange_multipliers, dtype=np.float64)
    loss_array = np.asarray(loss_terms, dtype=np.float64)
    ls_array = np.asarray(ls_terms, dtype=np.float64)
    restraint_array = np.asarray(restraint_terms, dtype=np.float64)
    target_array = np.asarray(target_total, dtype=np.float64)
    iteration_array = np.asarray(iterations, dtype=np.int32)
    converged_array = np.asarray(converged, dtype=bool)

    output_path = args.output or (
        ensure_results_dir(microstate_root.name, "onestepRESP") / "charges.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        charges=charges,
        initial_charges=initial,
        labels=labels_array,
        symbols=symbols_array,
        configs=config_array,
        rmse=rmse_array,
        rrms=rrms_array,
        sum_q=sum_array,
        lagrange=lambda_array,
        loss=loss_array,
        ls_term=ls_array,
        restraint=restraint_array,
        target_total_charge=target_array,
        iterations=iteration_array,
        pdb=str(pdb_path),
        geometry=geometry_meta,
        frame_index=args.frame,
        grid_frame_index=args.grid_frame,
        restraint_a=args.a,
        restraint_b=args.b,
        restraint_q0=args.q0,
        restrain_all_atoms=restrain_all,
        converged=converged_array,
        failed_configs=np.asarray(failed_configs, dtype=object),
    )
    print(f"Saved aggregated RESP charges to {output_path}")


if __name__ == "__main__":
    main()
