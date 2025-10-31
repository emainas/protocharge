from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml

from linearESPcharges.linear import prepare_linear_system
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


def load_total_charge(constraint_path: Path) -> float:
    """Read the total charge value from the total_constraint YAML."""
    with constraint_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("total_charge:"):
                return float(line.split(":", 1)[1].strip())
    raise ValueError(f"No total_charge entry found in {constraint_path}")


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
    total_charge: float,
    bucket_constraints: Iterable[Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (C, d) representing linear constraints in atom space."""
    atom_count, bucket_count = expansion_matrix.shape
    rows: List[np.ndarray] = []
    targets: List[float] = []

    rows.append(np.ones(atom_count, dtype=float))
    targets.append(total_charge)

    for constraint in bucket_constraints:
        bucket_index = int(constraint["bucket"])
        value = float(constraint["value"])
        if bucket_index < 0 or bucket_index >= bucket_count:
            raise IndexError(
                f"Constraint references bucket {bucket_index} but only {bucket_count} buckets available."
            )
        row = expansion_matrix[:, bucket_index].astype(float)
        if not np.any(row):
            raise ValueError(f"Bucket {bucket_index} has no associated atoms in expansion matrix.")
        rows.append(row)
        targets.append(value * float(row.sum()))

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
    reduced_design_matrix: np.ndarray,
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

    Ar = np.asarray(reduced_design_matrix, dtype=float)
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
        if exc.args:
            solution = exc.args[0]
        else:
            raise

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
) -> np.ndarray:
    """Solve the reduced least-squares problem with constraints projected into bucket space."""
    reduced_design_matrix = design_matrix @ expansion_matrix
    projected_constraints = constraint_matrix @ expansion_matrix

    H = reduced_design_matrix.T @ reduced_design_matrix
    g = reduced_design_matrix.T @ esp_values
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
    project_root = _project_root()
    symmetry_bucket_path = (
        project_root / "data" / "microstates" / "PPP" / "symmetry-buckets" / "r8.dat"
    )


    # STEP 1. Load the symmetry buckets from data/microstates/<microstate>/symmetry-buckets/...
    # Start with PPP as a first test case.
    # In total we have s=1,...,S symmetry buckets. S is going to be equal to the number of elements in
    # the loaded list of lists. S = len(np.loadtxt('data/microstates/PPP/symmetry-buckets/r8.dat', dtype=int))
    # print S to verify.
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
    resp_out = project_root / "data" / "raw" / "resp.out"
    esp_xyz = project_root / "data" / "raw" / "esp.xyz"
    design_matrix, esp_values, _, _ = prepare_linear_system(
        resp_out,
        esp_xyz,
        atom_count,
    )
    reduced_design_matrix = design_matrix @ P
    total_constraint_path = (
        project_root
        / "data"
        / "microstates"
        / "PPP"
        / "charge-contraints"
        / "total_constraint.yaml"
    )
    total_charge_target = load_total_charge(total_constraint_path)
    bucket_constraints_path = (
        project_root
        / "data"
        / "microstates"
        / "PPP"
        / "charge-contraints"
        / "bucket_constraints.yaml"
    )
    bucket_constraints = load_bucket_constraints(bucket_constraints_path)
    C, d = build_atom_constraint_system(P, total_charge_target, bucket_constraints)
    d_vector = d.flatten()
    theta_linear, lagrange_multipliers = solve_least_squares_with_constraints(
        design_matrix, esp_values, P, C, d
    )
    p_linear = P @ theta_linear
    total_charge_linear = float(p_linear.sum())
    np.set_printoptions(precision=6, suppress=True)
    print(f"Linear raw ESP charge vector p with shape {p_linear.shape}")
    print(p_linear)
    print(
        f"Total charge from p_linear: {total_charge_linear:.6f} "
        f"(target {total_charge_target:.6f})"
    )
    atom_labels = load_atom_labels_from_pdb(
        project_root / "data" / "microstates" / "PPP" / "ppp.pdb"
    )
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
    mask_step1_path = (
        project_root
        / "data"
        / "microstates"
        / "PPP"
        / "charge-contraints"
        / "mask_step_1.yaml"
    )
    restraint_mask_step1 = load_mask_from_yaml(
        mask_step1_path, atom_labels, symmetry_buckets
    )
    a_step1 = 0.0005
    b_step1 = 0.1
    print(f"Step 1 restraint mask shape: {restraint_mask_step1.shape}")
    print(f"Number of restrained atoms in step 1: {int(restraint_mask_step1.sum())}")
    print(f"Step 1 restraint parameters: a={a_step1}, b={b_step1}")

    logger_step1, theta_resp_step1, lambda_resp_step1 = resp_step(
        reduced_design_matrix,
        esp_values,
        P,
        atom_labels,
        C,
        d_vector,
        restraint_mask_step1,
        a_step1,
        b_step1,
        theta_linear,
        lagrange_multipliers,
        p_fixed=np.zeros(atom_count, dtype=float),
        description="RESP step one",
    )
    print(f"RESP step one evaluations logged: {len(logger_step1)}")

    if logger_step1 and plt is not None:
        kkt_values = [entry["kkt_norm"] for entry in logger_step1]
        iterations = range(1, len(kkt_values) + 1)
        plot_path = (
            project_root
            / "data"
            / "microstates"
            / "PPP"
            / "twostepRESP"
            / "resp_step1_loss.png"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
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

    mask_step2_path = (
        project_root
        / "data"
        / "microstates"
        / "PPP"
        / "charge-contraints"
        / "mask_step_2.yaml"
    )
    mask_step2_atom = load_mask_from_yaml(
        mask_step2_path, atom_labels, symmetry_buckets
    )
    if np.any(mask_step2_atom):
        mask_step2_flat = mask_step2_atom.flatten().astype(bool)
        bucket_variable = np.array(
            [any(atom_idx in bucket for atom_idx in np.where(mask_step2_flat)[0]) for bucket in symmetry_buckets],
            dtype=bool,
        )
        if not np.any(bucket_variable):
            raise ValueError("mask_step_2.yaml did not map to any symmetry buckets.")
        bucket_fixed = ~bucket_variable

        P_variable = P[:, bucket_variable]
        theta_var_init = theta_resp_step1[bucket_variable]
        theta_fixed_components = theta_resp_step1[bucket_fixed]

        design_variable = design_matrix @ P_variable
        esp_values_column = esp_values.reshape(-1, 1)
        if np.any(bucket_fixed):
            P_fixed = P[:, bucket_fixed]
            design_fixed = design_matrix @ P_fixed
            esp_adjusted = esp_values_column - design_fixed @ theta_fixed_components
            p_fixed = (P_fixed @ theta_fixed_components).flatten()
        else:
            esp_adjusted = esp_values_column
            p_fixed = np.zeros(atom_count, dtype=float)

        esp_adjusted = esp_adjusted.reshape(-1)
        d_adjusted = d_vector - (C @ p_fixed)

        a_step2 = 0.001
        b_step2 = 0.1
        logger_step2, theta_var_step2, lambda_resp_step2 = resp_step(
            design_variable,
            esp_adjusted,
            P_variable,
            atom_labels,
            C,
            d_adjusted,
            mask_step2_atom,
            a_step2,
            b_step2,
            theta_var_init,
            lambda_resp_step1,
            p_fixed=p_fixed,
            description="RESP step two",
        )
        print(f"RESP step two evaluations logged: {len(logger_step2)}")

        theta_resp_step2_full = theta_resp_step1.copy()
        theta_resp_step2_full[bucket_variable] = theta_var_step2
        theta_final = theta_resp_step2_full
        lambda_final = lambda_resp_step2
        if plt is not None and logger_step2:
            kkt_values_2 = [entry["kkt_norm"] for entry in logger_step2]
            iterations_2 = range(1, len(kkt_values_2) + 1)
            plot2_path = (
                project_root
                / "data"
                / "microstates"
                / "PPP"
                / "twostepRESP"
                / "resp_step2_loss.png"
            )
            plot2_path.parent.mkdir(parents=True, exist_ok=True)
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


    final_charges = P @ theta_final
    print(f"Final RESP total charge: {float(final_charges.sum()):+.6f}")


if __name__ == "__main__":
    main()
