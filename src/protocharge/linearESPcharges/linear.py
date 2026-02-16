from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

from protocharge.resp_parser import ParseRespDotOut, ParseESPXYZ

ANGSTROM_TO_BOHR = 1.8897261254578281

# ============================================================
# Units for ESP charge fitting
#
# - Atom coordinates (R) and grid coordinates (G):
#     must be in bohr (1 Å = 1.8897261254578281 bohr).
#
# - ESP values (V):
#     must be in atomic units of potential (Hartree/e).
#     TeraChem's esp.xyz already provides ESP in a.u.
#
# - Total charge (Q):
#     unitless (e.g., +1.0, 0.0, -1.0).
#
# - Coulomb matrix A:
#     A[i,j] = 1 / r_ij
#     where r_ij is distance (in bohr) between grid point i and atom j.
#     With these units, A @ q has the same units as V.
#
# - Charges (q):
#     unitless, results are in electrons.
#
# Summary:
#   → Convert all coordinates (Å → bohr).
#   → Use ESP from esp.xyz directly (already a.u.).
#   → Ensure Q is the correct integer/float.
# ============================================================


def build_design_matrix(
    grid_coordinates_bohr: np.ndarray,
    atom_positions_bohr: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    diffs = grid_coordinates_bohr[:, np.newaxis, :] - atom_positions_bohr[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    if np.any(distances <= epsilon):
        raise ValueError("Encountered grid point too close to an atom position when forming Coulomb matrix.")
    return 1.0 / distances


def _metrics(A: np.ndarray, V: np.ndarray, q: np.ndarray) -> Dict[str, Any]:
    pred = A @ q
    resid = pred - V
    rmse = float(np.sqrt(np.mean(resid**2)))
    rrms = float(np.sqrt(np.mean(resid**2) / np.mean(V**2))) if np.any(V) else float("nan")
    return {"pred": pred, "rmse": rmse, "rrms": rrms, "sum_q": float(q.sum())}

def _solve_sym(H: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(H, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(H, b, rcond=None)
        return x

@dataclass
class explicit_solution:
    """Closed-form LS + Lagrange projection (no block).
       q* = q0 - ((1^T q0 - Q) / (1^T H^{-1} 1)) * H^{-1} 1
       where H=A^T A, q0 = H^{-1}A^T V."""
    ridge: float = 0.0

    def fit(
        self,
        A: np.ndarray,
        V: np.ndarray,
        Q: float,
        *,
        constraint_mask: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        _, N = A.shape
        # Default to the classic "sum all charges" constraint if no mask is provided.
        constraint_vec = constraint_mask if constraint_mask is not None else np.ones(N)
        H = A.T @ A
        if self.ridge > 0.0:
            H = H + self.ridge * np.eye(N)
        g = A.T @ V
        q0 = _solve_sym(H, g)                  # unconstrained LS
        c  = _solve_sym(H, constraint_vec)     # correction direction
        alpha = float(constraint_vec @ c)
        s = float(constraint_vec @ q0)
        q = q0 - ((s - Q) / alpha) * c
        out = {"q": q, "q0": q0, "H": H, "g": g, "alpha": alpha, "s": s}
        out.update(_metrics(A, V, q))
        if constraint_mask is not None:
            out["sum_q_constraint"] = float(q @ constraint_mask)
        return out

def prepare_linear_system(
    resp_out: Path | str,
    esp_xyz: Path | str,
    number_of_atoms: int,
    *,
    frame_index: int | None = None,
    grid_frame_index: int = 0,
    return_positions: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray] | Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    resp_parser = ParseRespDotOut(resp_out, number_of_atoms)
    frames = resp_parser.extract_frames()
    grid_frames = ParseESPXYZ(esp_xyz).frames()

    if frame_index is None:
        frame_index = len(frames) - 1
    elif frame_index < 0:
        frame_index = len(frames) + frame_index

    frame = frames[frame_index]

    atom_positions_bohr = np.asarray(frame.positions, dtype=np.float64)
    grid_coordinates_angstrom = np.asarray(grid_frames[grid_frame_index].coordinates, dtype=np.float64)
    grid_coordinates_bohr = grid_coordinates_angstrom * ANGSTROM_TO_BOHR
    esp_values = np.asarray(grid_frames[grid_frame_index].potentials, dtype=np.float64)

    design_matrix = build_design_matrix(grid_coordinates_bohr, atom_positions_bohr)

    esp_charges = np.asarray(frame.esp_charges, dtype=np.float64)
    total_charge = float(esp_charges.sum())

    if return_positions:
        return design_matrix, esp_values, total_charge, esp_charges, atom_positions_bohr
    return design_matrix, esp_values, total_charge, esp_charges
