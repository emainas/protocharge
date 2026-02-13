from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import newton_krylov

try:  # SciPy >=1.14
    from scipy.optimize import NoConvergence  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from scipy.optimize.nonlin import NoConvergence  # type: ignore[attr-defined]

from biliresp.multiconfresp.mcresp import (
    ConfigurationSystem,
    _load_configuration_system,
    _microstate_root,
    _ordered_configurations,
    _project_root,
    _resolve_pdb_path,
    stack_configurations,
    _save_system,
)
from biliresp.paths import (
    ensure_results_dir,
    microstate_constraints_root,
    microstate_results_root,
)
from biliresp.twostepresp_masked_total.tsresp import (
    build_atom_constraint_system,
    build_expansion_matrix,
    build_total_constraint_mask,
    load_atom_labels_from_pdb,
    load_bucket_constraints,
    load_mask_from_yaml,
    load_symmetry_buckets,
    load_total_constraint,
    solve_least_squares_with_constraints,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiconfigurational RESP using a reduced_basic-space solver that enforces constraints exactly.",
    )
    parser.add_argument(
        "--microstate",
        required=True,
        help="Name of the microstate (expects data/microstates/<microstate> to exist).",
    )
    parser.add_argument(
        "--bucket-file",
        type=Path,
        help="Override symmetry bucket file (default: data/microstates/<microstate>/symmetry-buckets/r8.dat)",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        help="Override the reference PDB file used to infer the number of atoms.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="RESP frame index to extract from *.resp.out (default: last frame).",
    )
    parser.add_argument(
        "--grid-frame",
        type=int,
        default=0,
        help="ESP grid frame index inside *.esp.xyz (default: 0).",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        help="Optional upper bound on the number of configurations to load (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered configuration stems without constructing the stacked system.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional NPZ path to store the stacked system for reuse.",
    )
    parser.add_argument(
        "--input-dir-name",
        default="multiconfRESP",
        help="Subdirectory (under results/<microstate>) that stores Coulomb/ESP stacks.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="multiconfRESP_reduced_basic",
        help="Subdirectory (under results/<microstate>) where reduced_basic-space RESP outputs will be written.",
    )
    parser.add_argument(
        "--total-constraint",
        type=Path,
        help="Override total_constraint.yaml (default: configs/<microstate>/charge-contraints/total_constraint.yaml)",
    )
    parser.add_argument(
        "--bucket-constraints",
        type=Path,
        help="Override bucket_constraints.yaml (default: configs/<microstate>/charge-contraints/bucket_constraints.yaml)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist the stacked Coulomb matrix and ESP vector under input-dir-name.",
    )
    parser.add_argument(
        "--load-and-resp",
        action="store_true",
        help="Load precomputed Coulomb/ESP arrays and perform the reduced_basic RESP fit.",
    )
    parser.add_argument(
        "--svd-tol",
        type=float,
        default=1e-12,
        help="Cutoff used when computing the nullspace of the constraint matrix.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Optional quadratic regularisation applied to bucket charges (default: 0).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=400,
        help="Maximum Newton–Krylov iterations per reduced_basic RESP solve.",
    )
    parser.add_argument(
        "--f-tol",
        type=float,
        default=1e-12,
        help="Target residual tolerance passed to SciPy's newton_krylov solver.",
    )
    return parser.parse_args(argv)


def _save_stacked_matrices(
    microstate_root: Path,
    system,
    target_name: str,
) -> None:
    target_dir = ensure_results_dir(microstate_root.name, target_name)

    coulomb_path = target_dir / "coulomb_matrix.npz"
    esp_path = target_dir / "esp_vector.npz"

    np.savez_compressed(coulomb_path, A=system.design_matrix)
    np.savez_compressed(esp_path, Y=system.esp_values)

    print(f"Saved Coulomb matrix to {coulomb_path}")
    print(f"Saved ESP vector to {esp_path}")


def _load_saved_matrices(microstate_root: Path, source_name: str) -> Tuple[np.ndarray, np.ndarray]:
    target_dir = microstate_results_root(microstate_root.name) / source_name
    if not target_dir.is_dir():
        raise FileNotFoundError(
            f"Directory {target_dir} not found; run with --save first to generate matrices."
        )

    coulomb_path = target_dir / "coulomb_matrix.npz"
    esp_path = target_dir / "esp_vector.npz"
    if not coulomb_path.is_file():
        raise FileNotFoundError(f"Missing Coulomb matrix file: {coulomb_path}")
    if not esp_path.is_file():
        raise FileNotFoundError(f"Missing ESP vector file: {esp_path}")

    with np.load(coulomb_path) as data:
        if "A" not in data:
            raise KeyError(f"File {coulomb_path} does not contain dataset 'A'.")
        design_matrix = np.asarray(data["A"], dtype=float)

    with np.load(esp_path) as data:
        if "Y" not in data:
            raise KeyError(f"File {esp_path} does not contain dataset 'Y'.")
        esp_values = np.asarray(data["Y"], dtype=float).reshape(-1)

    return design_matrix, esp_values


def _serialize_logger(logger: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    def _convert(value: object) -> object:
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value

    return [{key: _convert(value) for key, value in entry.items()} for entry in logger]


def _save_resp_outputs(
    microstate_root: Path,
    target_name: str,
    charges_step1: np.ndarray,
    charges_final: np.ndarray,
    logger_step1: Sequence[Dict[str, float]],
    logger_step2: Sequence[Dict[str, float]],
) -> None:
    target_dir = ensure_results_dir(microstate_root.name, target_name)

    charges_step1_path = target_dir / "charges_step1.npy"
    charges_final_path = target_dir / "charges_final.npy"
    log1_path = target_dir / "resp_step1_log.json"
    log2_path = target_dir / "resp_step2_log.json"

    np.save(charges_step1_path, charges_step1)
    np.save(charges_final_path, charges_final)

    with log1_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize_logger(logger_step1), handle, indent=2)
    with log2_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize_logger(logger_step2), handle, indent=2)

    print(f"Saved RESP step one charges to {charges_step1_path}")
    print(f"Saved RESP final charges to {charges_final_path}")
    print(f"Saved RESP step one log to {log1_path}")
    print(f"Saved RESP step two log to {log2_path}")


def _nullspace_components(matrix: np.ndarray, rhs: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (theta_particular, Z, rank) solving matrix @ theta = rhs with nullspace basis."""
    if matrix.size == 0:
        return rhs.copy().reshape(-1, 1), np.zeros((rhs.size, 0), dtype=float), 0

    U, s, Vt = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(s > tol))
    Z = Vt.T[:, rank:]
    theta_p = np.linalg.lstsq(matrix, rhs, rcond=tol)[0]
    return theta_p.reshape(-1, 1), Z, rank


def _project_initial(theta_init: np.ndarray, theta_particular: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Project the initial guess onto the affine constraint space."""
    if Z.size == 0:
        return theta_particular.copy()
    delta = theta_init - theta_particular
    return theta_particular + Z @ (Z.T @ delta)


def _objective_components(
    design_matrix: np.ndarray,
    esp_values: np.ndarray,
    expansion_matrix: np.ndarray,
    mask_atom: np.ndarray,
    a: float,
    b: float,
    theta: np.ndarray,
    p_fixed: np.ndarray,
    ridge: float,
    constraint_bucket: np.ndarray,
    constraint_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return (gradient, constraint_residual, loss_linear, loss_restraint)."""
    linear_residual = design_matrix @ theta - esp_values.reshape(-1, 1)
    grad_linear = 2.0 * (design_matrix.T @ linear_residual)
    if ridge > 0.0:
        grad_linear = grad_linear + 2.0 * ridge * theta

    p = expansion_matrix @ theta + p_fixed.reshape(-1, 1)
    sqrt_term = np.sqrt(p * p + b * b)
    restraint_grad_atom = mask_atom * a * p / sqrt_term
    grad_restraint = expansion_matrix.T @ restraint_grad_atom

    grad = grad_linear + grad_restraint
    loss_linear = float(linear_residual.T @ linear_residual)
    if ridge > 0.0:
        loss_linear += float(ridge * (theta.T @ theta))
    loss_restraint = float(np.sum(mask_atom * a * (sqrt_term - b)))
    constraint_residual = constraint_bucket @ theta - constraint_targets.reshape(-1, 1)
    return grad, constraint_residual, loss_linear, loss_restraint


def _recover_lagrange_multipliers(
    constraint_bucket: np.ndarray,
    grad: np.ndarray,
    tol: float,
) -> np.ndarray:
    if constraint_bucket.size == 0:
        return np.zeros((0, 1), dtype=float)
    solution = np.linalg.lstsq(constraint_bucket.T, -grad, rcond=tol)[0]
    return solution.reshape(-1, 1)


def _run_reduced_basic_step(
    reduced_basic_design_matrix: np.ndarray,
    esp_values: np.ndarray,
    expansion_matrix: np.ndarray,
    constraint_matrix_atom: np.ndarray,
    constraint_targets: np.ndarray,
    mask_atom: np.ndarray,
    a: float,
    b: float,
    theta_init: np.ndarray,
    *,
    svd_tol: float,
    ridge: float,
    maxiter: int,
    f_tol: float,
    p_fixed: np.ndarray,
    description: str,
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    Ar = np.asarray(reduced_basic_design_matrix, dtype=float)
    V = np.asarray(esp_values, dtype=float).reshape(-1, 1)
    P = np.asarray(expansion_matrix, dtype=float)
    C_atom = np.asarray(constraint_matrix_atom, dtype=float)
    d_atom = np.asarray(constraint_targets, dtype=float).reshape(-1, 1)
    mask = np.asarray(mask_atom, dtype=float).reshape(-1, 1)
    theta0 = np.asarray(theta_init, dtype=float).reshape(-1, 1)
    p_fixed_vec = np.asarray(p_fixed, dtype=float).reshape(-1, 1)

    constraint_bucket = C_atom @ P
    theta_particular, Z, _ = _nullspace_components(constraint_bucket, d_atom, svd_tol)
    theta_projected = _project_initial(theta0, theta_particular, Z)

    logger: List[Dict[str, float]] = []

    if Z.size == 0:
        grad, constraint_residual, loss_linear, loss_restraint = _objective_components(
            Ar,
            V,
            P,
            mask,
            a,
            b,
            theta_particular,
            p_fixed_vec,
            ridge,
            constraint_bucket,
            d_atom,
        )
        total_loss = loss_linear + loss_restraint
        grad_norm = float(np.linalg.norm(grad))
        constraint_norm = float(np.linalg.norm(constraint_residual))
        kkt_norm = float(np.linalg.norm(constraint_residual))
        logger.append(
            {
                "eval": 1,
                "loss": total_loss,
                "loss_linear": loss_linear,
                "loss_restraint": loss_restraint,
                "grad_norm": grad_norm,
                "constraint_norm": constraint_norm,
                "kkt_norm": kkt_norm,
                "description": description,
            }
        )
        lambda_sol = _recover_lagrange_multipliers(constraint_bucket, grad, svd_tol)
        return logger, theta_particular, lambda_sol

    eval_counter = {"value": 0}

    def _log_metrics(theta: np.ndarray, grad: np.ndarray, constraint_residual: np.ndarray, loss_linear: float, loss_restraint: float) -> Dict[str, float]:
        eval_counter["value"] += 1
        total_loss = loss_linear + loss_restraint
        grad_norm = float(np.linalg.norm(grad))
        constraint_norm = float(np.linalg.norm(constraint_residual))
        projected_grad = Z.T @ grad
        kkt_vec = np.concatenate([projected_grad.reshape(-1, 1), constraint_residual], axis=0)
        kkt_norm = float(np.linalg.norm(kkt_vec))
        entry = {
            "eval": eval_counter["value"],
            "loss": total_loss,
            "loss_linear": loss_linear,
            "loss_restraint": loss_restraint,
            "grad_norm": grad_norm,
            "constraint_norm": constraint_norm,
            "kkt_norm": kkt_norm,
            "description": description,
        }
        return entry

    def residual(y_vec: np.ndarray) -> np.ndarray:
        theta = theta_particular + Z @ y_vec.reshape(-1, 1)
        grad, constraint_residual, loss_linear, loss_restraint = _objective_components(
            Ar,
            V,
            P,
            mask,
            a,
            b,
            theta,
            p_fixed_vec,
            ridge,
            constraint_bucket,
            d_atom,
        )
        logger.append(_log_metrics(theta, grad, constraint_residual, loss_linear, loss_restraint))
        projected_grad = Z.T @ grad
        return projected_grad.reshape(-1)

    y0 = Z.T @ (theta_projected - theta_particular)
    try:
        solution = newton_krylov(residual, y0.reshape(-1), maxiter=maxiter, f_tol=f_tol)
    except NoConvergence as exc:  # pragma: no cover - propagate best iterate
        solution = exc.args[0] if exc.args else y0.reshape(-1)

    theta_sol = theta_particular + Z @ solution.reshape(-1, 1)
    grad_fin, constraint_fin, loss_lin_fin, loss_rest_fin = _objective_components(
        Ar,
        V,
        P,
        mask,
        a,
        b,
        theta_sol,
        p_fixed_vec,
        ridge,
        constraint_bucket,
        d_atom,
    )
    logger.append(_log_metrics(theta_sol, grad_fin, constraint_fin, loss_lin_fin, loss_rest_fin))
    lambda_sol = _recover_lagrange_multipliers(constraint_bucket, grad_fin, svd_tol)
    return logger, theta_sol, lambda_sol


def _run_resp_from_saved(
    microstate_root: Path,
    *,
    pdb_override: Path | None,
    input_dir: str,
    output_dir: str,
    svd_tol: float,
    ridge: float,
    maxiter: int,
    f_tol: float,
    bucket_file: Path | None,
    total_constraint: Path | None,
    bucket_constraints_path: Path | None,
    mask_step1_path: Path | None,
    mask_step2_path: Path | None,
) -> None:
    design_matrix, esp_values = _load_saved_matrices(microstate_root, input_dir)

    atom_count = design_matrix.shape[1]
    if esp_values.size != design_matrix.shape[0]:
        raise ValueError(
            "ESP vector length does not match Coulomb matrix row count. "
            f"Got {esp_values.size} vs {design_matrix.shape[0]}."
        )

    pdb_path = _resolve_pdb_path(microstate_root, pdb_override)
    atom_labels = load_atom_labels_from_pdb(pdb_path)
    if len(atom_labels) != atom_count:
        raise ValueError(
            f"Atom label count ({len(atom_labels)}) does not match Coulomb matrix column count ({atom_count})."
        )

    bucket_file = bucket_file or (microstate_root / "symmetry-buckets" / "r8.dat")
    if not bucket_file.is_file():
        raise FileNotFoundError(f"Symmetry bucket file {bucket_file} not found.")
    symmetry_buckets = load_symmetry_buckets(bucket_file)
    expansion_matrix = build_expansion_matrix(symmetry_buckets)
    if expansion_matrix.shape[0] != atom_count:
        raise ValueError(
            "Expansion matrix row count does not match atom count. "
            f"Got {expansion_matrix.shape[0]} vs {atom_count}."
        )

    constraint_root = microstate_constraints_root(microstate_root.name)
    total_constraint_path = total_constraint or (constraint_root / "total_constraint.yaml")
    bucket_constraint_path = bucket_constraints_path or (constraint_root / "bucket_constraints.yaml")
    if not total_constraint_path.is_file():
        raise FileNotFoundError(f"Total charge constraint file {total_constraint_path} not found.")
    if not bucket_constraint_path.is_file():
        raise FileNotFoundError(f"Bucket constraint file {bucket_constraint_path} not found.")

    total_charge_target, total_constraint_labels = load_total_constraint(total_constraint_path)
    total_constraint_mask = build_total_constraint_mask(atom_labels, total_constraint_labels)
    bucket_constraints = load_bucket_constraints(bucket_constraint_path)
    constraint_matrix, constraint_targets = build_atom_constraint_system(
        expansion_matrix, total_charge_target, bucket_constraints, total_constraint_mask
    )
    constraint_targets_vector = constraint_targets.flatten()

    mask_step1_path = mask_step1_path or (constraint_root / "mask_step_1.yaml")
    mask_step2_path = mask_step2_path or (constraint_root / "mask_step_2.yaml")
    mask_step1 = load_mask_from_yaml(mask_step1_path, atom_labels, symmetry_buckets)
    mask_step2 = load_mask_from_yaml(mask_step2_path, atom_labels, symmetry_buckets)

    reduced_basic_design_matrix = design_matrix @ expansion_matrix
    theta_linear, _ = solve_least_squares_with_constraints(
        design_matrix,
        esp_values,
        expansion_matrix,
        constraint_matrix,
        constraint_targets,
    )

    a_step1 = 0.0005
    b_step1 = 0.1
    a_step2 = 0.001
    b_step2 = 0.1

    logger_step1, theta_step1, lambda_step1 = _run_reduced_basic_step(
        reduced_basic_design_matrix,
        esp_values,
        expansion_matrix,
        constraint_matrix,
        constraint_targets_vector,
        mask_step1,
        a_step1,
        b_step1,
        theta_linear,
        svd_tol=svd_tol,
        ridge=ridge,
        maxiter=maxiter,
        f_tol=f_tol,
        p_fixed=np.zeros(atom_count, dtype=float),
        description="Reduced RESP step one (ensemble)",
    )

    charges_step1 = (expansion_matrix @ theta_step1).flatten()
    total_charge_step1 = float(charges_step1.sum())

    logger_step2: List[Dict[str, float]] = []
    theta_final = theta_step1

    if np.any(mask_step2):
        mask2_bool = mask_step2.flatten().astype(bool)
        bucket_variable = np.array(
            [any(mask2_bool[atom_idx] for atom_idx in bucket) for bucket in symmetry_buckets],
            dtype=bool,
        )
        if np.any(bucket_variable):
            expansion_variable = expansion_matrix[:, bucket_variable]
            expansion_fixed = expansion_matrix[:, ~bucket_variable]

            theta_var_init = theta_step1[bucket_variable].reshape(-1, 1)
            theta_fixed = theta_step1[~bucket_variable].reshape(-1, 1)

            design_variable = design_matrix @ expansion_variable
            esp_values_column = esp_values.reshape(-1, 1)
            if np.any(~bucket_variable):
                design_fixed = design_matrix @ expansion_fixed
                esp_adjusted = esp_values_column - design_fixed @ theta_fixed
                p_fixed_vec = (expansion_fixed @ theta_fixed).flatten()
            else:
                esp_adjusted = esp_values_column
                p_fixed_vec = np.zeros(atom_count, dtype=float)

            esp_adjusted = esp_adjusted.reshape(-1)
            constraint_adjusted = constraint_targets_vector - (constraint_matrix @ p_fixed_vec)

            logger_step2, theta_variable, lambda_step2 = _run_reduced_basic_step(
                design_variable,
                esp_adjusted,
                expansion_variable,
                constraint_matrix,
                constraint_adjusted,
                mask_step2,
                a_step2,
                b_step2,
                theta_var_init,
                svd_tol=svd_tol,
                ridge=ridge,
                maxiter=maxiter,
                f_tol=f_tol,
                p_fixed=p_fixed_vec,
                description="Reduced RESP step two (ensemble)",
            )

            theta_full = theta_step1.copy()
            theta_full[bucket_variable, :] = theta_variable
            theta_final = theta_full
        else:
            print("mask_step_2.yaml did not select any symmetry buckets; skipping RESP step two.")
    else:
        print("mask_step_2.yaml is empty; skipping RESP step two.")

    charges_final = (expansion_matrix @ theta_final).flatten()
    total_charge_final = float(charges_final.sum())

    _save_resp_outputs(
        microstate_root,
        output_dir,
        charges_step1,
        charges_final,
        logger_step1,
        logger_step2,
    )

    print(f"Reduced RESP step one total charge: {total_charge_step1:+.6f}")
    print(f"Reduced RESP final total charge: {total_charge_final:+.6f}")
    print(f"Reduced RESP step one evaluations logged: {len(logger_step1)}")
    print(f"Reduced RESP step two evaluations logged: {len(logger_step2)}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    project_root = _project_root()
    microstate_root = _microstate_root(project_root, args.microstate)

    if args.load_and_resp:
        if args.save:
            raise ValueError("--save cannot be combined with --load-and-resp.")
        _run_resp_from_saved(
            microstate_root,
            pdb_override=args.pdb,
            input_dir=args.input_dir_name,
            output_dir=args.output_dir_name,
            svd_tol=args.svd_tol,
            ridge=args.ridge,
            maxiter=args.maxiter,
            f_tol=args.f_tol,
            bucket_file=args.bucket_file,
            total_constraint=args.total_constraint,
            bucket_constraints_path=args.bucket_constraints,
            mask_step1_path=args.mask_step1 if hasattr(args, "mask_step1") else None,
            mask_step2_path=args.mask_step2 if hasattr(args, "mask_step2") else None,
        )
        return

    resp_dir = microstate_root / "terachem" / "respout"
    esp_dir = microstate_root / "terachem" / "espxyz"
    if not resp_dir.is_dir():
        raise FileNotFoundError(f"Missing respout directory: {resp_dir}")
    if not esp_dir.is_dir():
        raise FileNotFoundError(f"Missing espxyz directory: {esp_dir}")

    configs = _ordered_configurations(resp_dir, esp_dir)
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    if not configs:
        raise FileNotFoundError(
            f"No matching resp.out / esp.xyz pairs found under {microstate_root}."
        )

    print(f"Found {len(configs)} configurations for microstate {args.microstate}.")
    if args.dry_run:
        for stem, resp_path, esp_path in configs:
            print(f"{stem}: {resp_path.name} | {esp_path.name}")
        return

    pdb_path = _resolve_pdb_path(microstate_root, args.pdb)
    atom_labels = load_atom_labels_from_pdb(pdb_path)
    number_of_atoms = len(atom_labels)

    config_systems: List[ConfigurationSystem] = []
    for idx, (stem, resp_path, esp_path) in enumerate(configs, start=1):
        system = _load_configuration_system(
            stem,
            resp_path,
            esp_path,
            number_of_atoms,
            frame_index=args.frame,
            grid_frame_index=args.grid_frame,
        )
        config_systems.append(system)
        if idx % 50 == 0 or idx == len(configs):
            print(f"Loaded {idx}/{len(configs)} configurations")

    ensemble = stack_configurations(config_systems)

    total_grid_points = ensemble.design_matrix.shape[0]
    atom_count = ensemble.design_matrix.shape[1]
    print(
        "Stacked design matrix shape: "
        f"{ensemble.design_matrix.shape} (grid points × atoms)."
    )
    print(
        f"Stacked ESP vector length: {ensemble.esp_values.size}; "
        f"atom count: {atom_count}."
    )
    charge_min = float(np.min(ensemble.total_charges))
    charge_max = float(np.max(ensemble.total_charges))
    print(
        f"Mean total charge: {ensemble.total_charge:+.6f} "
        f"(min {charge_min:+.6f}, max {charge_max:+.6f}); "
        f"aggregated grid points: {total_grid_points}."
    )

    if args.save:
        _save_stacked_matrices(microstate_root, ensemble, args.input_dir_name)

    if args.output is not None:
        _save_system(args.output, ensemble, labels=atom_labels)
        print(f"Saved stacked system to {args.output.resolve()}")


if __name__ == "__main__":
    main()
