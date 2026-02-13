from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biliresp.linearESPcharges.linear import prepare_linear_system, explicit_solution
from biliresp.resp_parser import ParseRespDotOut
from biliresp.resp.resp import fit_resp_charges, load_geometry_symbols

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RESP_OUT = DATA_DIR / "resp.out"
ESP_XYZ = DATA_DIR / "esp.xyz"
GEOM_XYZ = DATA_DIR / "1.pose.xyz"
NUMBER_OF_ATOMS = 78


def _load_reference_frame():
    frames = ParseRespDotOut(RESP_OUT, NUMBER_OF_ATOMS).extract_frames()
    if not frames:
        raise ValueError("No frames parsed from resp.out")
    frame = frames[-1]
    if not frame.resp_charges:
        raise ValueError("RESP charges missing in resp.out last frame")
    if not frame.esp_charges:
        raise ValueError("ESP unrestrained charges missing in resp.out last frame")
    return frame


def test_resp_solver_matches_terachem_last_frame():
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")

    project_reports = Path(__file__).resolve().parents[1] / "reports"
    project_reports.mkdir(parents=True, exist_ok=True)
    plot_path = project_reports / "resp_loss.png"
    result = fit_resp_charges(
        RESP_OUT,
        ESP_XYZ,
        GEOM_XYZ,
        NUMBER_OF_ATOMS,
        frame_index=-1,
        save_loss_plot=True,
        loss_plot_path=plot_path,
        restrain_all_atoms=True,
    )
    reference_frame = _load_reference_frame()
    expected = np.asarray(reference_frame.resp_charges, dtype=float)
    expected_esp = np.asarray(reference_frame.esp_charges, dtype=float)

    A, V, total_charge, _ = prepare_linear_system(
        RESP_OUT,
        ESP_XYZ,
        NUMBER_OF_ATOMS,
        frame_index=-1,
    )
    symbols = load_geometry_symbols(GEOM_XYZ, frame_index=-1)

    linear_solver = explicit_solution()
    linear_result = linear_solver.fit(A, V, total_charge)
    print(
        "Comparing linear solution",
        np.array2string(linear_result["q"][:3], precision=6, suppress_small=True, separator=", "),
        "with Terachem ESP",
        np.array2string(expected_esp[:3], precision=6, suppress_small=True, separator=", ")
    )

    print(
        "Comparing our charges",
        np.array2string(result["charges"][:3], precision=6, suppress_small=True, separator=", "),
        "with Terachem",
        np.array2string(expected[:3], precision=6, suppress_small=True, separator=", ")
    )

    diff = result["charges"] - expected
    max_abs_diff = float(np.max(np.abs(diff)))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(expected != 0.0, np.abs(diff / expected), np.nan)
    finite_rel = rel_diff[np.isfinite(rel_diff)]
    max_rel_diff = float(finite_rel.max()) if finite_rel.size else float("nan")

    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print(f"Max absolute difference: {max_abs_diff:.8f}")
    print(f"Max relative difference: {max_rel_diff:.8f}")
    print("ACTUAL:", np.array2string(result["charges"], precision=6, suppress_small=True, separator=", "))
    print("DESIRED:", np.array2string(expected, precision=6, suppress_small=True, separator=", "))
    np.testing.assert_allclose(result["charges"], expected, rtol=0.0, atol=1e-5)
    assert result["sum_q"] == pytest.approx(float(expected.sum()), abs=1e-5)
    assert np.all(result["mask"])
    assert result["loss_history"], "loss history should capture each solver iteration"
    assert result["loss_history"][-1] == pytest.approx(result["loss"], abs=1e-6)
    assert plot_path.exists(), "loss plot file was not created"
    np.testing.assert_allclose(linear_result["q"], expected_esp, rtol=0.0, atol=1e-5)
    assert linear_result["sum_q"] == pytest.approx(float(expected_esp.sum()), abs=1e-10)
