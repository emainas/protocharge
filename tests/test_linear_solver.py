from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from biliresp.linearESPcharges.linear import (
    explicit_solution,
    prepare_linear_system,
)
from biliresp.resp_parser import ParseRespDotOut


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RESP_OUT = DATA_DIR / "resp.out"
ESP_XYZ = DATA_DIR / "esp.xyz"
NUMBER_OF_ATOMS = 78


def test_linear_solvers_agree_and_reduce_residual():
    A, V, Q, resp_charges = prepare_linear_system(
        RESP_OUT,
        ESP_XYZ,
        NUMBER_OF_ATOMS,
        frame_index=-1,
    )

    expl = explicit_solution(ridge=0.0)

    res_expl = expl.fit(A, V, Q)

    # sanity: last unrestrained RESP charge first entry should match known value
    assert resp_charges[0] == pytest.approx(-0.517177, abs=1e-6)

    # Solvers should agree to numerical tolerance
    np.testing.assert_allclose(res_expl["sum_q"], Q, rtol=0.0, atol=1e-12)

    baseline_residual = A @ resp_charges - V
    baseline_rmse = float(np.sqrt(np.mean(baseline_residual**2)))
    baseline_rrms = float(np.sqrt(np.mean(baseline_residual**2) / np.mean(V**2)))
    assert res_expl["rmse"] <= baseline_rmse

    header = "atom  resp_charge   explicit      explicit-resp"
    print(header)
    for idx in range(78):
        print(
            f"{idx:4d}  {resp_charges[idx]:+11.6f}  {res_expl['q'][idx]:+11.6f}"
            f"  {res_expl['q'][idx]-resp_charges[idx]:+14.6f}"
        )

    print(
        f"\nΣq (explicit) = {res_expl['sum_q']:.12f}, RMSE = {res_expl['rmse']:.6e}, RRMS = {res_expl['rrms']:.6e}"
    )
    reported_rrms = _read_unrestrained_rrms(RESP_OUT, NUMBER_OF_ATOMS)
    print(
        f"Resp baseline Σq = {Q:.12f}, RMSE (calc) = {baseline_rmse:.6e}, RRMS (calc) = {baseline_rrms:.6e}, RRMS (reported) = {reported_rrms:.6e}"
    )


def _read_unrestrained_rrms(resp_path: Path, n_atoms: int) -> float:
    parser = ParseRespDotOut(resp_path, n_atoms)
    frames = parser.extract_frames()
    if not frames:
        raise ValueError("No frames parsed from resp.out")
    last = frames[-1]
    if last.esp_rms_error is None:
        raise ValueError("ESP RRMS not available for last frame")
    return float(last.esp_rms_error)
