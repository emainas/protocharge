# Multiconfigurational two step RESP fitting

# This will be similar to two step RESP fitting, but will fit multiple conformations at once,
# in the AMBER dyes paper this is called "ensemble fit". 

# We can do this by loading multiple conformations of the same molecule as well as their respective
# ESP grid calculations. 
# The conformations lie in:
# root/input/microstates/<microstate>/terachem/respout/conf<ID>.resp.out
# and the ESP grids lie in:
# root/input/microstates/<microstate>/terachem/espxyz/conf<ID>.esp.xyz
#
# 
# STEP 1: From all these confs and ESP grid measurements we will first load them for a specific microstate,
# that is passed through the command line argument --microstate <microstate>. 
# The math of how we define the matricies that will hold the data is the following:
'''
We consider \( N \) atomic monopoles with charges  
\[
\mathbf{q} = (q_1,\dots,q_N)^{\mathrm T} \in \mathbb{R}^N
\]  
located at nuclear positions  
\[
\mathbf{R} = (\mathbf{R}_1,\dots,\mathbf{R}_N)^{\mathrm T}
\]
with \( \mathbf{R}_n \in \mathbb{R}^3 \).  
From electronic structure calculations we obtain the quantum-mechanical electrostatic potential (ESP) at \( M \) grid points  
\[
\mathbf{r} = (\mathbf{r}_1,\dots,\mathbf{r}_M)^{\mathrm T}
\]
with \( \mathbf{r}_m \in \mathbb{R}^3 \).  
Our goal is to determine the optimal charge vector \( \mathbf{q} \) such that the classical Coulomb potential reproduces these ESP values.

---

### Classical Potential for a Single Grid Point

For a single grid point \( \mathbf{r}_m \), the classical potential is

\[
V(\mathbf{r}_m) 
= \sum_{n=1}^N \frac{q_n}{\lVert \mathbf{r}_m - \mathbf{R}_n \rVert}
= \sum_{n=1}^N q_n\, A(\mathbf{r}_m,\mathbf{R}_n),
\tag{B1}
\label{B1}
\]

where  
\[
A(\mathbf{r}_m,\mathbf{R}_n)=\frac{1}{\lVert \mathbf{r}_m - \mathbf{R}_n \rVert}
\]
is the free-space Green’s function of the three-dimensional Laplacian.

---

### Matrix–Vector Form

Stacking all \( M \) grid points yields the compact matrix–vector form

\[
\mathbf{V} = \mathbf{A}\,\mathbf{q}, 
\qquad 
A_{mn} = \frac{1}{\lVert \mathbf{r}_m - \mathbf{R}_n \rVert},
\tag{B2}
\label{B2}
\]

where \( \mathbf{A} \in \mathbb{R}^{M \times N} \) is the design (or Coulomb) matrix.  
Each element of \( \mathbf{A} \) is simply the Green’s function evaluated between a grid point \( m \) and an atomic site \( n \).

---

### Multi-Configuration System

For \( K \) molecular configurations, with corresponding design matrices \( \mathbf{A}^{(k)} \) and ESP measurement vectors \( \mathbf{Y}^{(k)} \), we form the block-stacked system

\[
\tilde{\mathbf{A}} =
\begin{bmatrix}
\mathbf{A}^{(1)} \\ \vdots \\ \mathbf{A}^{(K)}
\end{bmatrix}
\in \mathbb{R}^{\tilde{M}\times N},
\qquad
\tilde{\mathbf{Y}} =
\begin{bmatrix}
\mathbf{Y}^{(1)} \\ \vdots \\ \mathbf{Y}^{(K)}
\end{bmatrix}
\in \mathbb{R}^{\tilde{M}},
\tag{B3}
\label{B3}
\]

with  
\[
\tilde{M}=\sum_{k=1}^K M_k
\]
the total number of ESP points across all configurations.
'''

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from protocharge.training.linearESPcharges.linear import prepare_linear_system
from protocharge.paths import (
    ensure_results_dir,
    microstate_constraints_root,
    microstate_input_root,
    microstate_output_root,
    project_root,
)


def _project_root() -> Path:
    """Backward-compatible project root helper."""
    return project_root()
from protocharge.training.twostepresp_basic.tsresp import (
    build_atom_constraint_system,
    build_expansion_matrix,
    load_atom_labels_from_pdb,
    load_bucket_constraints,
    load_mask_from_yaml,
    load_symmetry_buckets,
    load_total_charge,
    resp_step,
    solve_least_squares_with_constraints,
)


@dataclass
class ConfigurationSystem:
    """Linear ESP system for a single conformation."""

    stem: str
    design_matrix: np.ndarray
    esp_values: np.ndarray
    total_charge: float
    esp_charges: np.ndarray
    atom_positions_bohr: np.ndarray


@dataclass
class EnsembleLinearSystem:
    """Block-stacked ESP system across multiple conformations."""

    design_matrix: np.ndarray
    esp_values: np.ndarray
    total_charge: float
    total_charges: np.ndarray
    config_slices: Dict[str, slice]
    config_order: List[str]
    esp_charges: np.ndarray
    atom_positions_bohr: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the multi-configuration RESP linear system for a microstate.",
    )
    parser.add_argument(
        "--microstate",
        required=True,
        help="Name of the microstate (expects input/microstates/<microstate> to exist).",
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
        "--save",
        action="store_true",
        help="Persist the stacked Coulomb matrix and ESP vector under output/<microstate>/multiconfRESP/.",
    )
    parser.add_argument(
        "--load-and-resp",
        action="store_true",
        help="Load precomputed Coulomb/ESP arrays and perform the multiconfiguration two-step RESP fit.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=400,
        help="Maximum Newton-Krylov iterations per RESP step (default: 400).",
    )
    return parser.parse_args(argv)


def _microstate_root(project_root: Path, microstate: str) -> Path:
    root = microstate_input_root(microstate)
    if not root.is_dir():
        raise NotADirectoryError(f"Microstate directory not found: {root}")
    return root


def _resolve_pdb_path(microstate_root: Path, override: Path | None) -> Path:
    if override is not None:
        resolved = override.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"PDB file {resolved} does not exist.")
        return resolved

    pdb_candidates = sorted(microstate_root.glob("*.pdb"))
    if not pdb_candidates:
        raise FileNotFoundError(
            f"No PDB file found under {microstate_root}; provide one via --pdb."
        )
    if len(pdb_candidates) > 1:
        raise RuntimeError(
            f"Multiple PDB files found under {microstate_root}; specify one with --pdb."
        )
    return pdb_candidates[0]


def _build_configuration_map(directory: Path, suffix: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        if not path.is_file():
            continue
        stem = path.name[: -len(suffix)]
        mapping[stem] = path
    return mapping


def _ordered_configurations(resp_dir: Path, esp_dir: Path) -> List[Tuple[str, Path, Path]]:
    resp_map = _build_configuration_map(resp_dir, ".resp.out")
    esp_map = _build_configuration_map(esp_dir, ".esp.xyz")
    common = sorted(resp_map.keys() & esp_map.keys())
    return [(stem, resp_map[stem], esp_map[stem]) for stem in common]


def _load_configuration_system(
    stem: str,
    resp_path: Path,
    esp_path: Path,
    number_of_atoms: int,
    *,
    frame_index: int,
    grid_frame_index: int,
) -> ConfigurationSystem:
    design_matrix, esp_values, total_charge, esp_charges, atom_positions = prepare_linear_system(
        resp_path,
        esp_path,
        number_of_atoms,
        frame_index=frame_index,
        grid_frame_index=grid_frame_index,
        return_positions=True,
    )
    return ConfigurationSystem(
        stem=stem,
        design_matrix=np.asarray(design_matrix, dtype=float),
        esp_values=np.asarray(esp_values, dtype=float).reshape(-1),
        total_charge=float(total_charge),
        esp_charges=np.asarray(esp_charges, dtype=float).reshape(-1),
        atom_positions_bohr=np.asarray(atom_positions, dtype=float),
    )


def stack_configurations(configs: Sequence[ConfigurationSystem]) -> EnsembleLinearSystem:
    if not configs:
        raise ValueError("No configurations provided for stacking.")

    atom_counts = {cfg.stem: cfg.esp_charges.size for cfg in configs}
    reference_atoms = next(iter(atom_counts.values()))
    for stem, count in atom_counts.items():
        if count != reference_atoms:
            raise ValueError(
                f"Configuration {stem} has {count} atoms but expected {reference_atoms}."
            )

    design_blocks = [cfg.design_matrix for cfg in configs]
    esp_blocks = [cfg.esp_values for cfg in configs]
    design_matrix = np.vstack(design_blocks)
    esp_values = np.concatenate(esp_blocks)

    total_charges = np.asarray([cfg.total_charge for cfg in configs], dtype=float)
    total_charge_mean = float(np.mean(total_charges))

    config_slices: Dict[str, slice] = {}
    start = 0
    for cfg in configs:
        stop = start + cfg.design_matrix.shape[0]
        config_slices[cfg.stem] = slice(start, stop)
        start = stop

    esp_charge_matrix = np.stack([cfg.esp_charges for cfg in configs], axis=1)
    positions = np.stack([cfg.atom_positions_bohr for cfg in configs], axis=0)

    return EnsembleLinearSystem(
        design_matrix=design_matrix,
        esp_values=esp_values,
        total_charge=total_charge_mean,
        total_charges=total_charges,
        config_slices=config_slices,
        config_order=[cfg.stem for cfg in configs],
        esp_charges=esp_charge_matrix,
        atom_positions_bohr=positions,
    )


def _save_system(path: Path, system: EnsembleLinearSystem, *, labels: Sequence[str]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        design_matrix=system.design_matrix,
        esp_values=system.esp_values,
        total_charge=np.array(system.total_charge, dtype=float),
        total_charges=system.total_charges,
        config_order=np.asarray(system.config_order, dtype=object),
        esp_charges=system.esp_charges,
        atom_positions_bohr=system.atom_positions_bohr,
        slices=np.asarray(
            [(stem, slc.start, slc.stop) for stem, slc in system.config_slices.items()],
            dtype=object,
        ),
        atom_labels=np.asarray(labels, dtype=object),
    )


def _save_stacked_matrices(
    microstate_root: Path,
    system: EnsembleLinearSystem,
) -> None:
    target_dir = ensure_results_dir(microstate_root.name, "multiconfRESP")

    coulomb_path = target_dir / "coulomb_matrix.npz"
    esp_path = target_dir / "esp_vector.npz"

    np.savez_compressed(coulomb_path, A=system.design_matrix)
    np.savez_compressed(esp_path, Y=system.esp_values)

    print(f"Saved Coulomb matrix to {coulomb_path}")
    print(f"Saved ESP vector to {esp_path}")


def _load_saved_matrices(microstate_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    target_dir = microstate_output_root(microstate_root.name) / "multiconfRESP"
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

    serialized: List[Dict[str, float]] = []
    for entry in logger:
        serialized.append({key: _convert(value) for key, value in entry.items()})
    return serialized


def _save_resp_outputs(
    microstate_root: Path,
    charges_step1: np.ndarray,
    charges_final: np.ndarray,
    logger_step1: Sequence[Dict[str, float]],
    logger_step2: Sequence[Dict[str, float]],
) -> None:
    target_dir = ensure_results_dir(microstate_root.name, "multiconfRESP")

    np.save(target_dir / "charges_step1.npy", charges_step1)
    np.save(target_dir / "charges_final.npy", charges_final)

    log1_path = target_dir / "resp_step1_log.json"
    log2_path = target_dir / "resp_step2_log.json"
    with (target_dir / "resp_step1_log.json").open("w", encoding="utf-8") as handle:
        json.dump(_serialize_logger(logger_step1), handle, indent=2)
    with (target_dir / "resp_step2_log.json").open("w", encoding="utf-8") as handle:
        json.dump(_serialize_logger(logger_step2), handle, indent=2)

    print(f"Saved RESP step one charges to {target_dir / 'charges_step1.npy'}")
    print(f"Saved RESP final charges to {target_dir / 'charges_final.npy'}")
    print(f"Saved RESP step one log to {log1_path}")
    print(f"Saved RESP step two log to {log2_path}")


def _run_resp_from_saved(
    microstate_root: Path,
    *,
    pdb_override: Optional[Path],
    maxiter: int,
) -> None:
    design_matrix, esp_values = _load_saved_matrices(microstate_root)

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

    bucket_file = microstate_root / "symmetry-buckets" / "r8.dat"
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
    total_constraint_path = constraint_root / "total_constraint.yaml"
    if not total_constraint_path.is_file():
        raise FileNotFoundError(f"Total charge constraint file {total_constraint_path} not found.")
    bucket_constraint_path = constraint_root / "bucket_constraints.yaml"
    if not bucket_constraint_path.is_file():
        raise FileNotFoundError(f"Bucket constraint file {bucket_constraint_path} not found.")

    total_charge_target = load_total_charge(total_constraint_path)
    bucket_constraints = load_bucket_constraints(bucket_constraint_path)
    constraint_matrix, constraint_targets = build_atom_constraint_system(
        expansion_matrix, total_charge_target, bucket_constraints
    )
    constraint_targets_vector = constraint_targets.flatten()

    reduced_basic_design_matrix = design_matrix @ expansion_matrix
    theta_linear, lambda_linear = solve_least_squares_with_constraints(
        design_matrix,
        esp_values,
        expansion_matrix,
        constraint_matrix,
        constraint_targets,
    )

    mask_step1_path = constraint_root / "mask_step_1.yaml"
    mask_step2_path = constraint_root / "mask_step_2.yaml"
    mask_step1 = load_mask_from_yaml(mask_step1_path, atom_labels, symmetry_buckets)
    mask_step2 = load_mask_from_yaml(mask_step2_path, atom_labels, symmetry_buckets)

    a_step1 = 0.0005
    b_step1 = 0.1
    a_step2 = 0.001
    b_step2 = 0.1

    logger_step1, theta_step1, lambda_step1 = resp_step(
        reduced_basic_design_matrix,
        esp_values,
        expansion_matrix,
        atom_labels,
        constraint_matrix,
        constraint_targets_vector,
        mask_step1,
        a_step1,
        b_step1,
        theta_linear,
        lambda_linear,
        maxiter=maxiter,
        p_fixed=np.zeros(atom_count, dtype=float),
        description="RESP step one (ensemble)",
        print_summary=False,
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
                p_fixed = (expansion_fixed @ theta_fixed).flatten()
            else:
                esp_adjusted = esp_values_column
                p_fixed = np.zeros(atom_count, dtype=float)

            esp_adjusted = esp_adjusted.reshape(-1)
            constraint_adjusted = constraint_targets_vector - (constraint_matrix @ p_fixed)

            logger_step2, theta_variable, lambda_step2 = resp_step(
                design_variable,
                esp_adjusted,
                expansion_variable,
                atom_labels,
                constraint_matrix,
                constraint_adjusted,
                mask_step2,
                a_step2,
                b_step2,
                theta_var_init,
                lambda_step1,
                maxiter=maxiter,
                p_fixed=p_fixed,
                description="RESP step two (ensemble)",
                print_summary=False,
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
        charges_step1,
        charges_final,
        logger_step1,
        logger_step2,
    )

    print(f"RESP step one total charge: {total_charge_step1:+.6f}")
    print(f"RESP final total charge: {total_charge_final:+.6f}")
    print(f"RESP step one evaluations logged: {len(logger_step1)}")
    print(f"RESP step two evaluations logged: {len(logger_step2)}")

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    root_dir = project_root()
    microstate_root = _microstate_root(root_dir, args.microstate)

    if args.load_and_resp:
        if args.save:
            raise ValueError("--save cannot be combined with --load-and-resp.")
        _run_resp_from_saved(
            microstate_root,
            pdb_override=args.pdb,
            maxiter=args.maxiter,
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
        _save_stacked_matrices(microstate_root, ensemble)

    if args.output is not None:
        _save_system(args.output, ensemble, labels=atom_labels)
        print(f"Saved stacked system to {args.output.resolve()}")


if __name__ == "__main__":
    main()
