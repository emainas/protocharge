import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from protocharge.constants.atomic_masses import atomic_masses
from protocharge.resp_parser import ParseDotXYZ, ParseRespDotOut

BOHR_PER_ANG = 1.8897261254578281
DEBYE_PER_E_BOHR = 2.541746

# --- helpers ---
def _dipole_from_charges(q: np.ndarray, R_bohr: np.ndarray, origin_bohr: np.ndarray) -> Tuple[np.ndarray, float]:
    """μ = Σ q_A (R_A - origin). Returns (vec_D, mag_D)."""
    Rc = R_bohr - origin_bohr[None, :]
    mu_vec_e_bohr = (q[:, None] * Rc).sum(axis=0)
    mu_vec_D = DEBYE_PER_E_BOHR * mu_vec_e_bohr
    return mu_vec_D, float(np.linalg.norm(mu_vec_D))

def _normalize_frame_index(frame_index: int, total: int) -> int:
    if total == 0:
        raise ValueError("No frames available in resp output")
    if frame_index is None:
        frame_index = -1
    if frame_index < 0:
        frame_index = total + frame_index
    if not 0 <= frame_index < total:
        raise IndexError(f"frame_index {frame_index} out of range for {total} frames")
    return frame_index


def center_of_mass_bohr_from_xyz(
    xyz_path: str | Path,
    frame_index: int = -1,
    *,
    coords: np.ndarray | None = None,
    coords_unit: str = "ang",
) -> np.ndarray:
    """Mass-weighted center of mass in bohr using element labels from an xyz file.

    ``coords`` lets callers supply any coordinates (e.g., from RESP output) while
    still reusing the element-to-mass mapping from the xyz file.
    """

    frames = ParseDotXYZ(xyz_path).elements()
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("No frames available in xyz file")
    try:
        idx = _normalize_frame_index(frame_index, total_frames)
    except IndexError:
        idx = _normalize_frame_index(-1, total_frames)

    frame = frames[idx]
    symbols = frame.symbols
    n_atoms = len(symbols)

    if coords is None:
        coords_ang = np.asarray(frame.coordinates, dtype=float)
    else:
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.shape != (n_atoms, 3):
            raise ValueError(
                f"Custom coordinates shape {coords_arr.shape} does not match number of atoms ({n_atoms})"
            )
        unit = coords_unit.lower()
        if unit in {"ang", "angs", "angstrom", "angstroms"}:
            coords_ang = coords_arr
        elif unit in {"bohr", "a0", "au"}:
            coords_ang = coords_arr / BOHR_PER_ANG
        else:
            raise ValueError("coords_unit must be 'ang' (angstrom) or 'bohr'")

    try:
        masses = np.array([atomic_masses[symbol] for symbol in symbols], dtype=float)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Atomic mass for element '{missing}' not found in atomic_masses dictionary") from exc

    total_mass = masses.sum()
    if total_mass == 0.0:
        raise ValueError("Total mass computed as zero; check atomic_masses dictionary")

    com_ang = (masses[:, None] * coords_ang).sum(axis=0) / total_mass
    return com_ang * BOHR_PER_ANG

