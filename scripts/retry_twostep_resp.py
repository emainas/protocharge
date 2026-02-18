from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from protocharge.paths import microstate_constraints_root

from protocharge.linearESPcharges.linear import prepare_linear_system
from protocharge.twostepresp_basic_basic.tsresp import (
    build_atom_constraint_system,
    build_expansion_matrix,
    load_atom_labels_from_pdb,
    load_bucket_constraints,
    load_mask_from_yaml,
    load_symmetry_buckets,
    resp_step,
    solve_least_squares_with_constraints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry the two-step RESP fitting for a single configuration with custom iteration limits."
        )
    )
    parser.add_argument(
        "--microstate-root",
        type=Path,
        required=True,
        help="Path to the microstate directory (e.g., input/microstates/PPP).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf2371",
        help="Configuration stem without extension (default: conf2371).",
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
        "--step1-maxiter",
        type=int,
        default=120,
        help="Maximum Newton-Krylov iterations for the first RESP step (default: 120).",
    )
    parser.add_argument(
        "--step2-maxiter",
        type=int,
        default=240,
        help="Maximum Newton-Krylov iterations for the second RESP step (default: 240).",
    )
    parser.add_argument(
        "--bucket-file",
        type=Path,
        help="Override symmetry-bucket file (default: <microstate-root>/symmetry-buckets/r8.dat).",
    )
    parser.add_argument(
        "--total-constraint",
        type=Path,
        help="Override total charge constraint YAML (default: configs/<microstate>/charge-contraints/total_constraint.yaml).",
    )
    parser.add_argument(
        "--bucket-constraints",
        type=Path,
        help="Override bucket constraint YAML (default: configs/<microstate>/charge-contraints/bucket_constraints.yaml).",
    )
    parser.add_argument(
        "--mask-step1",
        type=Path,
        help="Override mask YAML for step 1 (default: configs/<microstate>/charge-contraints/mask_step_1.yaml).",
    )
    parser.add_argument(
        "--mask-step2",
        type=Path,
        help="Override mask YAML for step 2 (default: configs/<microstate>/charge-contraints/mask_step_2.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional NPZ output path for the fitted charges.",
    )
    parser.add_argument(
        "--show-charges",
        action="store_true",
        help="Print per-atom charges for each step.",
    )
    return parser.parse_args()


def find_pdb(root: Path) -> Path:
    pdbs = sorted(root.glob("*.pdb"))
    if not pdbs:
        raise FileNotFoundError(f"No PDB file found under {root}")
    if len(pdbs) > 1:
        raise RuntimeError("Multiple PDB files found; please specify one explicitly.")
    return pdbs[0]


def main() -> None:
    args = parse_args()

    microstate_root = args.microstate_root.resolve()
    if not microstate_root.is_dir():
        raise NotADirectoryError(f"{microstate_root} is not a directory")

    stem = args.config
    resp_path = microstate_root / "terachem" / "respout" / f"{stem}.resp.out"
    esp_path = microstate_root / "terachem" / "espxyz" / f"{stem}.esp.xyz"
    if not resp_path.is_file():
        raise FileNotFoundError(f"RESP output {resp_path} not found.")
    if not esp_path.is_file():
        raise FileNotFoundError(f"ESP grid {esp_path} not found.")

    pdb_path = find_pdb(microstate_root)
    atom_labels = load_atom_labels_from_pdb(pdb_path)
    n_atoms = len(atom_labels)

    bucket_file = (args.bucket_file or microstate_root / "symmetry-buckets" / "r8.dat").resolve()
    if not bucket_file.is_file():
        raise FileNotFoundError(f"Symmetry bucket file {bucket_file} not found.")
    symmetry_buckets = load_symmetry_buckets(bucket_file)
    P = build_expansion_matrix(symmetry_buckets)

    constraint_root = microstate_constraints_root(microstate_root.name)
    total_constraint_path = (
        args.total_constraint or (constraint_root / "total_constraint.yaml")
    )
    total_constraint_path = total_constraint_path.resolve()
    if not total_constraint_path.is_file():
        raise FileNotFoundError(f"Total constraint file {total_constraint_path} not found.")

    bucket_constraint_path = (
        args.bucket_constraints or (constraint_root / "bucket_constraints.yaml")
    )
    bucket_constraint_path = bucket_constraint_path.resolve()
    if not bucket_constraint_path.is_file():
        raise FileNotFoundError(f"Bucket constraint file {bucket_constraint_path} not found.")

    bucket_constraints = load_bucket_constraints(bucket_constraint_path)

    mask_step1 = load_mask_from_yaml(
        (args.mask_step1 or (constraint_root / "mask_step_1.yaml")).resolve(),
        atom_labels,
        symmetry_buckets,
    )
    mask_step2 = load_mask_from_yaml(
        (args.mask_step2 or (constraint_root / "mask_step_2.yaml")).resolve(),
        atom_labels,
        symmetry_buckets,
    )

    design_matrix, esp_values, total_charge_cfg, _ = prepare_linear_system(
        resp_path,
        esp_path,
        n_atoms,
        frame_index=args.frame,
        grid_frame_index=args.grid_frame,
    )

    C, d = build_atom_constraint_system(P, total_charge_cfg, bucket_constraints)
    d_vector = d.flatten()

    theta_linear, lambda_linear = solve_least_squares_with_constraints(
        design_matrix,
        esp_values,
        P,
        C,
        d,
    )

    logger1, theta_step1, lambda_step1 = resp_step(
        design_matrix @ P,
        esp_values,
        P,
        atom_labels,
        C,
        d_vector,
        mask_step1,
        a=0.0005,
        b=0.1,
        theta_init=theta_linear,
        lambda_init=lambda_linear,
        maxiter=args.step1_maxiter,
        p_fixed=np.zeros(n_atoms, dtype=float),
        description=f"{stem} – RESP step one",
        print_summary=args.show_charges,
    )
    resp_step1_charges = (P @ theta_step1).flatten()

    logger2: List[dict] = []
    resp_step2_charges = resp_step1_charges
    theta_full = theta_step1
    lambda_step2 = lambda_step1

    if np.any(mask_step2):
        mask2_bool = mask_step2.flatten().astype(bool)
        bucket_variable = np.array(
            [any(mask2_bool[atom_idx] for atom_idx in bucket) for bucket in symmetry_buckets],
            dtype=bool,
        )
        P_variable = P[:, bucket_variable]
        P_fixed = P[:, ~bucket_variable]

        theta_var_init = theta_step1[bucket_variable]
        theta_fixed = theta_step1[~bucket_variable]

        design_var = design_matrix @ P_variable
        esp_values_col = esp_values.reshape(-1, 1)
        if np.any(~bucket_variable):
            design_fix = design_matrix @ P_fixed
            esp_adjusted = esp_values_col - design_fix @ theta_fixed
            p_fixed = (P_fixed @ theta_fixed).flatten()
        else:
            esp_adjusted = esp_values_col
            p_fixed = np.zeros(n_atoms, dtype=float)

        esp_adjusted = esp_adjusted.reshape(-1)
        d_adjusted = d_vector - (C @ p_fixed)

        try:
            logger2, theta_var, lambda_step2 = resp_step(
                design_var,
                esp_adjusted,
                P_variable,
                atom_labels,
                C,
                d_adjusted,
                mask_step2,
                a=0.001,
                b=0.1,
                theta_init=theta_var_init,
                lambda_init=lambda_step1,
                maxiter=args.step2_maxiter,
                p_fixed=p_fixed,
                description=f"{stem} – RESP step two",
                print_summary=args.show_charges,
            )

            theta_full = theta_step1.copy()
            theta_full[bucket_variable] = theta_var
            resp_step2_charges = (P @ theta_full).flatten()
        except (ValueError, RuntimeError) as exc:
            print(
                f"[WARN] {stem}: second RESP step failed ({exc}); keeping step-one charges."
            )

    final_step1 = logger1[-1] if logger1 else None
    final_step2 = logger2[-1] if logger2 else None
    if final_step1:
        print(
            f"Step one completed in {final_step1['eval']} evaluations "
            f"(kkt_norm={final_step1['kkt_norm']:.3e})."
        )
    if final_step2:
        print(
            f"Step two completed in {final_step2['eval']} evaluations "
            f"(kkt_norm={final_step2['kkt_norm']:.3e})."
        )
    elif np.any(mask_step2):
        print("Step two did not converge; step-one charges retained.")

    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = microstate_root / "twostepRESP_basic" / f"{stem}_retry.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        labels=np.asarray([stem], dtype=object),
        step1=resp_step1_charges.reshape(-1, 1),
        step2=resp_step2_charges.reshape(-1, 1),
        theta_full=theta_full.flatten(),
        lambda_step2=lambda_step2.flatten(),
    )
    print(f"Saved fitted charges to {output_path}")


if __name__ == "__main__":
    main()
