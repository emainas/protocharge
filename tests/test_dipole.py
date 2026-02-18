from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from protocharge.utils.dipole import (
    BOHR_PER_ANG,
    _dipole_from_charges,
    _normalize_frame_index,
    center_of_mass_bohr_from_xyz,
)
from protocharge.training.resp_parser import ParseDotXYZ, ParseRespDotOut
from protocharge.training.linearESPcharges.linear import explicit_solution, prepare_linear_system

DATA_DIR = Path(__file__).resolve().parents[1] / "input" / "raw"
RESP_OUT = DATA_DIR / "resp.out"
ESP_XYZ = DATA_DIR / "esp.xyz"
GEOM_XYZ = DATA_DIR / "1.pose.xyz"
NUMBER_OF_ATOMS = 78


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


def three_dipoles_last_frame(
    resp_out_path: Path,
    xyz_path: Path,
    R_bohr_lastframe: np.ndarray,
    q_opt: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    return _three_dipoles_for_frame(
        resp_out_path,
        xyz_path,
        R_bohr_lastframe,
        q_opt,
        frame_index=-1,
    )


def test_three_dipoles_for_frame():
    A, V, Q, resp_charges, coords_bohr = prepare_linear_system(
        RESP_OUT,
        ESP_XYZ,
        NUMBER_OF_ATOMS,
        frame_index=-1,
        return_positions=True,
    )

    solver = explicit_solution(ridge=0.0)
    res = solver.fit(A, V, Q)

    frame_index = -1
    dipoles = _three_dipoles_for_frame(
        RESP_OUT,
        GEOM_XYZ,
        coords_bohr,
        res["q"],
        frame_index=frame_index,
    )

    # Basic sanity: magnitudes and vectors exist
    for key in (
        "qm_dipole_vec_D",
        "qm_dipole_mag_D",
        "terachem_dipole_vec_D",
        "terachem_dipole_mag_D",
        "lagrange_dipole_vec_D",
        "lagrange_dipole_mag_D",
    ):
        assert key in dipoles

    # Optimized dipole should be extremely close to ESP-unrestrained dipole
    np.testing.assert_allclose(
        dipoles["lagrange_dipole_vec_D"],
        dipoles["terachem_dipole_vec_D"],
        atol=5e-4,
    )
    assert dipoles["lagrange_dipole_mag_D"] == pytest.approx(
        dipoles["terachem_dipole_mag_D"], abs=5e-4
    )

    # Both classical dipoles should be within ~0.1 Debye of the QM dipole
    assert dipoles["terachem_dipole_mag_D"] == pytest.approx(
        dipoles["qm_dipole_mag_D"], abs=1e-1
    )
    assert dipoles["lagrange_dipole_mag_D"] == pytest.approx(
        dipoles["qm_dipole_mag_D"], abs=1e-1
    )

    # Ensure COM estimates are close (RESP log vs mass-weighted)
    com_resp = dipoles["COM_bohr_resp"]
    com_mass = dipoles["COM_bohr_mass"]
    np.testing.assert_allclose(com_resp, com_mass, atol=1e-2)

    print("Mass-weighted COM (angstrom):", com_mass / BOHR_PER_ANG)

    print("QM dipole vector (Debye):", dipoles["qm_dipole_vec_D"])
    print("QM |μ| (Debye): {:.6f}".format(dipoles["qm_dipole_mag_D"]))
    print("Terachem dipole vector (Debye):", dipoles["terachem_dipole_vec_D"])
    print("Terachem |μ| (Debye): {:.6f}".format(dipoles["terachem_dipole_mag_D"]))
    print("Lagrange dipole vector (Debye):", dipoles["lagrange_dipole_vec_D"])
    print("Lagrange |μ| (Debye): {:.6f}".format(dipoles["lagrange_dipole_mag_D"]))


def test_center_of_mass_bohr_from_xyz_matches_length():
    # Default path uses coordinates from the xyz file
    com_bohr_default = center_of_mass_bohr_from_xyz(GEOM_XYZ, frame_index=-1)
    assert com_bohr_default.shape == (3,)

    # Custom coordinates in Angstrom should yield the same center
    geom_frames = ParseDotXYZ(GEOM_XYZ).elements()
    coords_ang = np.asarray(geom_frames[0].coordinates, dtype=float)
    com_bohr_custom = center_of_mass_bohr_from_xyz(
        GEOM_XYZ,
        frame_index=-1,
        coords=coords_ang,
        coords_unit="ang",
    )
    np.testing.assert_allclose(com_bohr_default, com_bohr_custom, atol=1e-12)
