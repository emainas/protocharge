from __future__ import annotations

import ast
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml

from biliresp.linearESPcharges.linear import prepare_linear_system
from biliresp.paths import ensure_results_dir, microstate_constraints_root
from scipy.optimize import newton_krylov
try:  # SciPy >=1.14
    from scipy.optimize import NoConvergence  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from scipy.optimize.nonlin import NoConvergence  # type: ignore[attr-defined]

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    plt = None

def _project_root() -> Path:
    """Return the project root relative to this source file."""
    return Path(__file__).resolve().parents[2]


def load_symmetry_buckets(path: Path) -> List[List[int]]:
    """Load the symmetry bucket definitions as a list of lists of atom indices."""
    with path.open("r", encoding="utf-8") as bucket_file:
        raw_content = bucket_file.read()

    buckets = ast.literal_eval(raw_content)
    if not isinstance(buckets, list):
        raise ValueError(f"Symmetry bucket file {path} did not contain a list.")

    normalized_buckets: List[List[int]] = []
    for bucket in buckets:
        if not isinstance(bucket, list):
            raise ValueError(f"Encountered non-list bucket {bucket!r} in {path}.")
        normalized_buckets.append([int(atom_index) for atom_index in bucket])
    return normalized_buckets


def build_expansion_matrix(symmetry_buckets: List[List[int]]) -> np.ndarray:
    """Construct the NxS expansion matrix P for the provided buckets."""
    if not symmetry_buckets:
        raise ValueError("Symmetry buckets cannot be empty.")

    highest_atom_index = max(max(bucket) for bucket in symmetry_buckets if bucket)
    atom_count = highest_atom_index + 1
    bucket_count = len(symmetry_buckets)

    expansion_matrix = np.zeros((atom_count, bucket_count), dtype=int)

    assigned_atoms = set()
    for bucket_index, bucket in enumerate(symmetry_buckets):
        for atom_index in bucket:
            if atom_index in assigned_atoms:
                raise ValueError(
                    f"Atom index {atom_index} assigned to multiple symmetry buckets."
                )
            expansion_matrix[atom_index, bucket_index] = 1
            assigned_atoms.add(atom_index)

    return expansion_matrix


def build_group_mask_from_indices(atom_count: int, indices: Iterable[int]) -> np.ndarray:
    """Return an atom-space mask with ones at the provided indices."""
    mask = np.zeros(atom_count, dtype=float)
    for idx in indices:
        if idx < 0 or idx >= atom_count:
            raise IndexError(f"Constraint index {idx} out of range for {atom_count} atoms.")
        mask[idx] = 1.0
    if mask.sum() == 0:
        raise ValueError("Group constraint indices did not select any atoms.")
    return mask.reshape(-1, 1)


def load_group_constraints(constraint_path: Path, atom_count: int) -> Tuple[List[np.ndarray], List[float]]:
    """Load group constraints (indices + target) from YAML."""
    if not constraint_path.exists():
        raise FileNotFoundError(f"Group constraint file {constraint_path} not found.")

    data = yaml.safe_load(constraint_path.read_text(encoding="utf-8")) or []
    if isinstance(data, dict) and "group_constraints" in data:
        groups_raw = data["group_constraints"]
    else:
        groups_raw = data

    if not isinstance(groups_raw, list):
        raise ValueError("Group constraint file must be a YAML list of groups.")

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
        mask = build_group_mask_from_indices(atom_count, indices)
        masks.append(mask)
        targets.append(float(entry["group_charge"]))

    if not masks:
        raise ValueError("No group constraints found in file.")
    return masks, targets


def load_frozen_buckets(constraint_path: Path) -> Dict[int, float]:
    """Load frozen bucket values from YAML (same format as bucket_constraints.yaml)."""
    if not constraint_path.exists():
        raise FileNotFoundError(f"Frozen constraint file {constraint_path} not found.")
    data = yaml.safe_load(constraint_path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict) and "bucket_constraints" in data:
        entries = data["bucket_constraints"]
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError("Frozen constraint file must be a YAML list of bucket entries.")
    frozen: Dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Bucket entry must be a mapping; got {entry!r}")
        if "bucket" not in entry or "value" not in entry:
            raise ValueError("Each bucket entry must have 'bucket' and 'value'.")
        idx = int(entry["bucket"])
        frozen[idx] = float(entry["value"])
    return frozen


def load_bucket_constraints(
    constraint_path: Path,
) -> List[Dict[str, float]]:
    """Parse bucket-specific charge constraints from YAML."""
    constraints: List[Dict[str, float]] = []
    current: Dict[str, float] = {}

    with constraint_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("- bucket:"):
                if current:
                    constraints.append(current)
                    current = {}
                bucket_index = int(line.split(":", 1)[1].strip())
                current["bucket"] = bucket_index
            elif line.startswith("value:"):
                value = float(line.split(":", 1)[1].strip())
                current["value"] = value
            # labels are informational only, so we can skip them for constraint building

        if current:
            constraints.append(current)

    return constraints


def build_atom_constraint_system(
    expansion_matrix: np.ndarray,
    group_masks: Sequence[np.ndarray],
    group_targets: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (C, d) representing linear constraints in atom space (groups only)."""
    atom_count, _ = expansion_matrix.shape
    rows: List[np.ndarray] = []
    targets: List[float] = []

    if len(group_masks) != len(group_targets):
        raise ValueError("group_masks and group_targets must have the same length.")

    for mask, target in zip(group_masks, group_targets):
        mask_vec = np.asarray(mask, dtype=float).reshape(-1)
        if mask_vec.size != atom_count:
            raise ValueError(
                f"Constraint mask length ({mask_vec.size}) does not match atom count ({atom_count})."
            )
        rows.append(mask_vec)
        targets.append(float(target))

    C = np.vstack(rows)
    d = np.array(targets, dtype=float).reshape(-1, 1)
    return C, d


def load_atom_labels_from_pdb(path: Path) -> List[str]:
    """Extract atom labels from a PDB file in their original ordering."""
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                labels.append(line[12:16].strip())
    if not labels:
        raise ValueError(f"No atom labels extracted from {path}")
    return labels


def load_mask_from_yaml(
    path: Path,
    atom_labels: Sequence[str],
    symmetry_buckets: Sequence[Sequence[int]],
) -> np.ndarray:
    """Load a mask configuration from YAML and return an Nx1 mask vector."""
    if not path.exists():
        raise FileNotFoundError(f"Mask configuration file {path} not found.")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []

    if isinstance(data, dict):
        raise ValueError(
            f"Mask configuration {path} should be a YAML list of atom labels; found mapping instead."
        )
    if not isinstance(data, list):
        raise ValueError(
            f"Mask configuration {path} must be a YAML list of atom labels or indices."
        )

    mask = np.zeros(len(atom_labels), dtype=float)
    for entry in data:
        if isinstance(entry, str):
            matched = False
            for idx, atom_label in enumerate(atom_labels):
                if atom_label == entry:
                    mask[idx] = 1.0
                    matched = True
            if not matched:
                raise ValueError(
                    f"Mask entry {entry!r} not found among atom labels."
                )
        elif isinstance(entry, int):
            if entry < 0 or entry >= len(atom_labels):
                raise IndexError(
                    f"Mask references atom index {entry} but only {len(atom_labels)} atoms are available."
                )
            mask[entry] = 1.0
        else:
            raise TypeError(
                f"Mask entry {entry!r} must be a string label or integer index."
            )

    return mask.reshape(-1, 1)


def resp_step(
    reduced_basic_design_matrix: np.ndarray,
    esp_values: np.ndarray,
    expansion_matrix: np.ndarray,
    atom_labels: Sequence[str],
    constraint_matrix_atom: np.ndarray,
    constraint_targets: np.ndarray,
    mask_atom: np.ndarray,
    a: float,
    b: float,
    theta_init: np.ndarray,
    lambda_init: np.ndarray,
    *,
    maxiter: int = 60,
    p_fixed: np.ndarray | None = None,
    description: str = "RESP step",
    print_summary: bool = True,
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """Perform a RESP step using Newton–Krylov on bucket variables."""

    Ar = np.asarray(reduced_basic_design_matrix, dtype=float)
    V = np.asarray(esp_values, dtype=float).reshape(-1)
    P = np.asarray(expansion_matrix, dtype=float)
    C_atom = np.asarray(constraint_matrix_atom, dtype=float)
    d = np.asarray(constraint_targets, dtype=float).reshape(-1)
    mask = np.asarray(mask_atom, dtype=float).reshape(-1)
    if p_fixed is None:
        p_fixed = np.zeros(P.shape[0], dtype=float)
    else:
        p_fixed = np.asarray(p_fixed, dtype=float).reshape(-1)

    theta0 = np.asarray(theta_init, dtype=float).reshape(-1)
    lambda0 = np.asarray(lambda_init, dtype=float).reshape(-1)

    Cr = C_atom @ P

    logger: List[Dict[str, float]] = []
    eval_count = {"value": 0}

    def residual(vec: np.ndarray) -> np.ndarray:
        eval_count["value"] += 1
        theta = vec[: theta0.size]
        lam = vec[theta0.size :]

        linear_residual = Ar @ theta - V
        grad_linear = 2.0 * (Ar.T @ linear_residual)

        p = P @ theta + p_fixed
        sqrt_term = np.sqrt(p * p + b * b)
        grad_restraint = P.T @ (mask * a * p / sqrt_term)

        grad = grad_linear + grad_restraint + Cr.T @ lam
        constraint_residual = Cr @ theta - d

        loss_linear = float(linear_residual @ linear_residual)
        loss_restraint = float(np.sum(mask * a * (sqrt_term - b)))
        total_loss = loss_linear + loss_restraint

        kkt_residual = np.concatenate([grad, constraint_residual])

        logger.append(
            {
                "eval": eval_count["value"],
                "loss": total_loss,
                "loss_linear": loss_linear,
                "loss_restraint": loss_restraint,
                "grad_norm": float(np.linalg.norm(grad)),
                "constraint_norm": float(np.linalg.norm(constraint_residual)),
                "kkt_norm": float(np.linalg.norm(kkt_residual)),
            }
        )

        return kkt_residual

    x0 = np.concatenate([theta0, lambda0])
    try:
        solution = newton_krylov(residual, x0, maxiter=maxiter, f_tol=1e-12)
    except NoConvergence as exc:  # pragma: no cover - propagate with logging information
        raise RuntimeError(f"{description} failed to converge") from exc

    solution = np.asarray(solution, dtype=float)
    theta_sol = solution[: theta0.size].reshape(-1, 1)
    lambda_sol = solution[theta0.size :].reshape(-1, 1)

    final_p = P @ theta_sol + p_fixed.reshape(-1, 1)
    total_charge = float(final_p.sum())
    if print_summary:
        print(f"{description} charges (atom label, charge):")
        for label, charge in zip(atom_labels, final_p.flatten()):
            print(f"{label:>8}: {charge:+.6f}")
        print(f"{description} total charge: {total_charge:+.6f}")

    return logger, theta_sol, lambda_sol


def solve_least_squares_with_constraints(
    design_matrix: np.ndarray,
    esp_values: np.ndarray,
    expansion_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_targets: np.ndarray,
    *,
    ridge: float = 0.0,
) -> np.ndarray:
    """Solve the reduced_basic least-squares problem with constraints projected into bucket space."""
    reduced_basic_design_matrix = design_matrix @ expansion_matrix
    projected_constraints = constraint_matrix @ expansion_matrix

    H = reduced_basic_design_matrix.T @ reduced_basic_design_matrix
    if ridge > 0.0:
        H = H + ridge * np.eye(H.shape[0])
    g = reduced_basic_design_matrix.T @ esp_values
    if g.ndim == 1:
        g = g.reshape(-1, 1)

    # Build the KKT system by augmenting H and g with the constraints.
    # We need the zero block for the lower right block of the KKT matrix,
    # because the constraints do not interact with each other in this formulation.
    zero_block = np.zeros((projected_constraints.shape[0], projected_constraints.shape[0]))
    lhs = np.block([[H, projected_constraints.T], [projected_constraints, zero_block]])
    rhs = np.vstack([g, constraint_targets])

    # Solve the KKT system with numpy's LAPACK implementation. 
    solution = np.linalg.solve(lhs, rhs)

    # The solution vector contains both the independent charges and the Lagrange multipliers.
    theta = solution[: expansion_matrix.shape[1]].reshape(-1, 1)
    lagrange_multipliers = solution[expansion_matrix.shape[1] :].reshape(-1, 1)
    return theta, lagrange_multipliers


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-step RESP (group constraints with frozen buckets, by index)")
    parser.add_argument("--microstate", required=True, help="Microstate name under data/microstates/<microstate>")
    parser.add_argument(
        "--bucket-file",
        type=Path,
        help="Override symmetry bucket file (default: data/microstates/<microstate>/symmetry-buckets/r8.dat)",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        help="Override PDB path (default: data/microstates/<microstate>/<microstate>.pdb)",
    )
    parser.add_argument(
        "--resp-out",
        type=Path,
        help="Override resp.out path (default: data/raw/resp.out)",
    )
    parser.add_argument(
        "--esp-xyz",
        type=Path,
        help="Override esp.xyz path (default: data/raw/esp.xyz)",
    )
    parser.add_argument(
        "--group-constraint",
        type=Path,
        help="Override group_constraint.yaml (default: configs/<microstate>/charge-contraints/group_constraint.yaml)",
    )
    parser.add_argument(
        "--frozen-buckets",
        type=Path,
        help="Override frozen.yaml (bucket list) (default: configs/<microstate>/charge-contraints/frozen.yaml)",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Optional quadratic regularisation on the linear solve (default: 0.0).",
    )
    args = parser.parse_args()

    project_root = _project_root()
    micro_root = project_root / "data" / "microstates" / args.microstate

    symmetry_bucket_path = args.bucket_file or (micro_root / "symmetry-buckets" / "r8.dat")
    pdb_path = args.pdb or (micro_root / f"{args.microstate}.pdb")
    resp_out = args.resp_out or (project_root / "data" / "raw" / "resp.out")
    esp_xyz = args.esp_xyz or (project_root / "data" / "raw" / "esp.xyz")
    constraint_root = microstate_constraints_root(args.microstate)
    group_constraint_path = args.group_constraint or (constraint_root / "group_constraint.yaml")
    frozen_path = args.frozen_buckets or (constraint_root / "frozen.yaml")

    if not symmetry_bucket_path.is_file():
        raise FileNotFoundError(f"Symmetry bucket file not found: {symmetry_bucket_path}")
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if not resp_out.is_file():
        raise FileNotFoundError(f"RESP output not found: {resp_out}")
    if not esp_xyz.is_file():
        raise FileNotFoundError(f"ESP XYZ not found: {esp_xyz}")
    if not group_constraint_path.is_file():
        raise FileNotFoundError(f"Group constraint file not found: {group_constraint_path}")
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Frozen constraint file not found: {frozen_path}")

    # STEP 1. Load the symmetry buckets from data/microstates/<microstate>/symmetry-buckets/...
    symmetry_buckets = load_symmetry_buckets(symmetry_bucket_path)
    S = len(symmetry_buckets)
    print(f"Loaded {S} symmetry buckets from {symmetry_bucket_path}")

    # STEP 2. Based on the different buckets, define the independent charge vector theta
    # that has dimension Sx1.
    theta = np.zeros((S, 1))

    # STEP 3. Define the expansion matrix P of dimension NxS that maps the independent
    # charges to the full charge vector. Here N is the total number of atoms in the
    # system.
    # The matrix P is defined such that each row corresponds to an atom in the system
    # and each column corresponds to a symmetry bucket. The entries of P are defined as:
    #   P[n,s] = 1 if atom n belongs to symmetry bucket s
    #            0 otherwise, n = 1,...,N; s = 1,...,S.
    P = build_expansion_matrix(symmetry_buckets)
    print(f"Constructed expansion matrix P with shape {P.shape}")

    # STEP 4. The full charge vector is p = P * theta that has dimension Nx1.
    p = P @ theta
    print(f"Computed full charge vector p has shape {p.shape}")

    # STEP 5. Calculate linear raw ESP charges by solving the linear system but with vector
    # p as defined above instead of a full charge vector q as we did in previous work.
    atom_count = P.shape[0]
    atom_labels = load_atom_labels_from_pdb(pdb_path)
    design_matrix, esp_values, _, _ = prepare_linear_system(
        resp_out,
        esp_xyz,
        atom_count,
    )
    group_masks, group_targets = load_group_constraints(group_constraint_path, len(atom_labels))
    frozen_map = load_frozen_buckets(frozen_path)

    bucket_count = P.shape[1]
    frozen_mask = np.zeros(bucket_count, dtype=bool)
    frozen_values: List[float] = []
    frozen_columns: List[int] = []
    for idx in range(bucket_count):
        if idx in frozen_map:
            frozen_mask[idx] = True
            frozen_columns.append(idx)
            frozen_values.append(frozen_map[idx])
    variable_mask = ~frozen_mask
    if not np.any(variable_mask):
        raise ValueError("All buckets are frozen; nothing to optimize.")

    P_frozen = P[:, frozen_mask]
    P_variable = P[:, variable_mask]
    theta_frozen = np.array(frozen_values, dtype=float).reshape(-1, 1)
    p_frozen = P_frozen @ theta_frozen if theta_frozen.size else np.zeros(atom_count, dtype=float).reshape(-1, 1)

    design_variable = design_matrix @ P_variable
    esp_adjusted = esp_values.reshape(-1, 1) - design_matrix @ p_frozen

    C, d = build_atom_constraint_system(P, group_masks, group_targets)
    d_adjusted = d - (C @ p_frozen)
    d_vector = d_adjusted.flatten()

    theta_linear, lagrange_multipliers = solve_least_squares_with_constraints(
        design_matrix, esp_adjusted, P_variable, C, d_adjusted, ridge=args.ridge
    )
    p_linear = P_variable @ theta_linear + p_frozen
    total_charge_linear = float(p_linear.sum())
    group_sums_linear = [float(mask.flatten() @ p_linear) for mask in group_masks]
    np.set_printoptions(precision=6, suppress=True)
    print(f"Linear raw ESP charge vector p with shape {p_linear.shape}")
    print(p_linear)
    print(f"Total charge from p_linear: {total_charge_linear:.6f}")
    for idx, (target, actual) in enumerate(zip(group_targets, group_sums_linear), start=1):
        print(f"Group {idx}: target={target:+.6f}  actual={actual:+.6f}")
    if len(atom_labels) != p_linear.size:
        raise ValueError(
            f"Atom label count {len(atom_labels)} does not match charge vector length {p_linear.size}"
        )
    print("Atom label / raw ESP charge pairs:")
    for label, charge in zip(atom_labels, p_linear.flatten()):
        print(f"{label:>8}: {charge:+.6f}")

    print(f"Lagrange multipliers from linear solve with shape {lagrange_multipliers.shape}")
    print(lagrange_multipliers)
    print(f"Constraint matrix C shape: {C.shape}")

    # STEP 7. Define the restraint potential with a mask vector of dimension Nx1. Define
    # the parameter a and the parameter b as usual. This time, we will mask over the heavy atoms
    # and leave the hydrogens unmasked.
    mask_step1_path = args.mask_step1 or (constraint_root / "mask_step_1.yaml")
    restraint_mask_step1 = load_mask_from_yaml(mask_step1_path, atom_labels, symmetry_buckets)
    a_step1 = 0.0005
    b_step1 = 0.1
    print(f"Step 1 restraint mask shape: {restraint_mask_step1.shape}")
    print(f"Number of restrained atoms in step 1: {int(restraint_mask_step1.sum())}")
    print(f"Step 1 restraint parameters: a={a_step1}, b={b_step1}")

    logger_step1, theta_resp_step1, lambda_resp_step1 = resp_step(
        design_variable,
        esp_adjusted.flatten(),
        P_variable,
        atom_labels,
        C,
        d_vector,
        restraint_mask_step1,
        a_step1,
        b_step1,
        theta_linear,
        lagrange_multipliers,
        p_fixed=p_frozen.flatten(),
        description="RESP step one",
    )
    print(f"RESP step one evaluations logged: {len(logger_step1)}")

    if logger_step1 and plt is not None:
        kkt_values = [entry["kkt_norm"] for entry in logger_step1]
        iterations = range(1, len(kkt_values) + 1)
        plot_path = (
            ensure_results_dir(args.microstate, "twostepRESP_frozen_buckets")
            / "resp_step1_loss.png"
        )
        plt.figure(figsize=(6, 4))
        plt.plot(iterations, kkt_values, marker="o", linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("KKT residual norm")
        plt.title("RESP Step 1 KKT Residual Progression")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved RESP step one KKT residual plot to {plot_path}")
    elif logger_step1:
        print("Matplotlib not available; skipping RESP step one KKT residual plot.")

    # STEP 8-10. Perform second RESP step on selected atoms if provided.
    theta_final = theta_resp_step1.copy()
    lambda_final = lambda_resp_step1.copy()

    mask_step2_path = args.mask_step2 or (constraint_root / "mask_step_2.yaml")
    mask_step2_atom = load_mask_from_yaml(mask_step2_path, atom_labels, symmetry_buckets)
    if np.any(mask_step2_atom):
        mask_step2_flat = mask_step2_atom.flatten().astype(bool)
        bucket_selected = np.array(
            [any(atom_idx in bucket for atom_idx in np.where(mask_step2_flat)[0]) for bucket in symmetry_buckets],
            dtype=bool,
        )
        if not np.any(bucket_selected):
            raise ValueError("mask_step_2.yaml did not map to any symmetry buckets.")
        bucket_variable_full = bucket_selected & variable_mask
        if not np.any(bucket_variable_full):
            print("mask_step_2.yaml only touched frozen buckets; skipping RESP step two.")
            logger_step2 = []
            theta_final = theta_resp_step1
            lambda_final = lambda_resp_step1
        else:
            bucket_variable_varspace = bucket_variable_full[variable_mask]

            P_variable_step2 = P[:, bucket_variable_full]

            theta_var_init = theta_resp_step1[bucket_variable_varspace]
            theta_fixed_components = theta_resp_step1[~bucket_variable_varspace]

            p_fixed_vec = (
                p_frozen
                + (P[:, variable_mask][:, ~bucket_variable_varspace] @ theta_fixed_components)
            )
            design_variable_step2 = design_matrix @ P_variable_step2
            esp_adjusted_step2 = esp_values.reshape(-1, 1) - design_matrix @ p_fixed_vec

            esp_adjusted_step2 = esp_adjusted_step2.reshape(-1)
            d_adjusted = d_vector - (C @ p_fixed_vec.flatten())

            a_step2 = 0.001
            b_step2 = 0.1
            logger_step2, theta_var_step2, lambda_resp_step2 = resp_step(
                design_variable_step2,
                esp_adjusted_step2,
                P_variable_step2,
                atom_labels,
                C,
                d_adjusted,
                mask_step2_atom,
                a_step2,
                b_step2,
                theta_var_init,
                lambda_resp_step1,
                p_fixed=p_fixed_vec.flatten(),
                description="RESP step two",
            )
            print(f"RESP step two evaluations logged: {len(logger_step2)}")

            theta_full_varspace = theta_resp_step1.copy()
            theta_full_varspace[bucket_variable_varspace] = theta_var_step2
            theta_final = theta_full_varspace
            lambda_final = lambda_resp_step2
            if plt is not None and logger_step2:
                kkt_values_2 = [entry["kkt_norm"] for entry in logger_step2]
                iterations_2 = range(1, len(kkt_values_2) + 1)
                plot2_path = (
                    ensure_results_dir(args.microstate, "twostepRESP_frozen_buckets")
                    / "resp_step2_loss.png"
                )
                plt.figure(figsize=(6, 4))
                plt.plot(iterations_2, kkt_values_2, marker="o", linewidth=1.5)
                plt.xlabel("Iteration")
                plt.ylabel("KKT residual norm")
                plt.title("RESP Step 2 KKT Residual Progression")
                plt.tight_layout()
                plt.savefig(plot2_path, dpi=200)
                plt.close()
                print(f"Saved RESP step two KKT residual plot to {plot2_path}")
            elif logger_step2:
                print("Matplotlib not available; skipping RESP step two KKT residual plot.")
    else:
        print("No atoms specified in mask_step_2.yaml; skipping RESP step two.")
        theta_final = theta_resp_step1
        lambda_final = lambda_resp_step1


    final_charges = P[:, variable_mask] @ theta_final + p_frozen
    print(f"Final RESP total charge: {float(final_charges.sum()):+.6f}")


if __name__ == "__main__":
    main()
