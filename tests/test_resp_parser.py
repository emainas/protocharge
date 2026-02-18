from __future__ import annotations

from pathlib import Path

import pytest

from protocharge.resp_parser import ParseRespDotOut, ParseESPXYZ, ParseDotXYZ


DATA_DIR = Path(__file__).resolve().parents[1] / "input" / "raw"


def test_extract_frames():
    parser = ParseRespDotOut(DATA_DIR / "resp.out", 78)
    assert parser.success_check()

    frames = parser.extract_frames()

    assert frames, "Expected at least one frame"

    first_frame = frames[0]

    assert len(first_frame.positions) == 78
    assert len(first_frame.esp_charges) == 78
    assert len(first_frame.exposure_fractions) == 78
    assert len(first_frame.resp_charges) == 78

    assert first_frame.positions[-1] == pytest.approx((29.349334, 44.452024, 48.860755))
    assert first_frame.esp_charges[-1] == pytest.approx(0.439305, abs=1e-6)
    assert first_frame.resp_charges[-1] == pytest.approx(0.44013, abs=1e-6)

    assert sum(first_frame.esp_charges) == pytest.approx(1.0, abs=1e-5)
    assert sum(first_frame.resp_charges) == pytest.approx(1.0, abs=1e-5)


def test_parse_esp_xyz():
    frames = ParseESPXYZ(DATA_DIR / "esp.xyz").frames()

    assert frames, "Expected ESP grid frames from esp.xyz"

    first_frame = frames[0]
    assert len(first_frame.coordinates) == len(first_frame.potentials)
    assert len(first_frame.coordinates) == 11486
    assert first_frame.coordinates[0] == pytest.approx((21.7902980, 23.20555, 23.4869096))
    assert first_frame.potentials[0] == pytest.approx(0.164433438524, abs=1e-12)


def test_parse_dot_xyz_elements():
    elements_frames = ParseDotXYZ(DATA_DIR / "1.pose.xyz").elements()

    assert elements_frames, "Expected at least one elements frame"

    first_elements = elements_frames[0]
    assert len(first_elements.symbols) == 78
    assert len(first_elements.coordinates) == 78
    assert first_elements.symbols[0] == "N"
    assert first_elements.coordinates[0] == pytest.approx((20.747, 23.133, 21.972))
