from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from protocharge.training.linearESPcharges.linear import explicit_solution, prepare_linear_system
from protocharge.training.resp_parser import ParseRespDotOut
from protocharge.utils.dipole import (
    BOHR_PER_ANG,
    _dipole_from_charges,
    _normalize_frame_index,
    center_of_mass_bohr_from_xyz,
)


def _load_yaml(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping: {path}")
    return data


def _three_dipoles_for_frame(
    resp_out_path: Path,
    xyz_path: Path,
    R_bohr_frame: np.ndarray,
    q_opt: np.ndarray,
    *,
    frame_index: int = -1,
) -> Dict[str, np.ndarray | float]:
    q_opt = np.asarray(q_opt, dtype=float)
    R_bohr_frame = np.asarray(R_bohr_frame, dtype=float)
    if R_bohr_frame.shape[0] != q_opt.shape[0]:
        raise ValueError("Number of coordinates and charges must match")

    parser = ParseRespDotOut(resp_out_path, q_opt.shape[0])
    frames = parser.extract_frames()
    idx = _normalize_frame_index(frame_index, len(frames))
    frame = frames[idx]

    if frame.center_of_mass is None:
        raise ValueError("CENTER OF MASS data missing in resp.out for selected frame")
    if frame.dipole_moment_vector is None or frame.dipole_moment_magnitude is None:
        raise ValueError("DIPOLE MOMENT data missing in resp.out for selected frame")

    com_bohr_resp = np.asarray(frame.center_of_mass, dtype=float) * BOHR_PER_ANG
    qm_vec = np.asarray(frame.dipole_moment_vector, dtype=float)
    qm_mag = float(frame.dipole_moment_magnitude)

    q_terachem = np.asarray(frame.esp_charges, dtype=float)
    R_resp = np.asarray(frame.positions, dtype=float)
    if q_terachem.shape[0] != q_opt.shape[0]:
        raise ValueError("RESP and optimized charge arrays have different lengths")

    terachem_vec, terachem_mag = _dipole_from_charges(q_terachem, R_resp, com_bohr_resp)
    lagrange_vec, lagrange_mag = _dipole_from_charges(q_opt, R_bohr_frame, com_bohr_resp)

    com_bohr_mass = center_of_mass_bohr_from_xyz(
        xyz_path,
        frame_index=frame_index,
        coords=R_bohr_frame,
        coords_unit="bohr",
    )

    return {
        "qm_dipole_vec_D": qm_vec,
        "qm_dipole_mag_D": qm_mag,
        "terachem_dipole_vec_D": terachem_vec,
        "terachem_dipole_mag_D": terachem_mag,
        "lagrange_dipole_vec_D": lagrange_vec,
        "lagrange_dipole_mag_D": lagrange_mag,
        "delta_terachem_vs_qm_vec_D": terachem_vec - qm_vec,
        "delta_terachem_vs_qm_mag_D": terachem_mag - qm_mag,
        "delta_lagrange_vs_qm_vec_D": lagrange_vec - qm_vec,
        "delta_lagrange_vs_qm_mag_D": lagrange_mag - qm_mag,
        "COM_bohr_resp": com_bohr_resp,
        "COM_bohr_mass": com_bohr_mass,
    }


def run_dipole_validation(config_path: Path) -> Dict[str, np.ndarray | float]:
    cfg = _load_yaml(config_path)
    resp_out = Path(cfg["resp_out"])
    esp_xyz = Path(cfg["esp_xyz"])
    geom_xyz = Path(cfg["geom_xyz"])
    n_atoms = int(cfg["n_atoms"])
    frame_index = int(cfg.get("frame", -1))

    A, V, Q, resp_charges, coords_bohr = prepare_linear_system(
        resp_out,
        esp_xyz,
        n_atoms,
        frame_index=frame_index,
        return_positions=True,
    )

    solver = explicit_solution()
    fit_result = solver.fit(A, V, Q)

    dipoles = _three_dipoles_for_frame(
        resp_out,
        geom_xyz,
        coords_bohr,
        fit_result["q"],
        frame_index=frame_index,
    )

    return dipoles
