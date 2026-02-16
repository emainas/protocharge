from __future__ import annotations

from pathlib import Path

from protocharge.terachem_processing import process_tc_resp_runs


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_process_tc_resp_runs(tmp_path: Path) -> None:
    microstate = tmp_path / "data" / "microstates" / "HID"
    conf_root = microstate / "input_tc_structures" / "confs"

    _write(conf_root / "conf1" / "nowater.rst7", "RST7\n")
    _write(conf_root / "conf1" / "resp.out", "RESP1\n")
    _write(conf_root / "conf1" / "scr.nowater" / "esp.xyz", "ESP1\n")

    _write(conf_root / "conf2" / "nowater.rst7", "RST7-2\n")
    _write(conf_root / "conf2" / "resp.out", "RESP2\n")
    _write(conf_root / "conf2" / "scr.nowater.rst7" / "esp.xyz", "ESP2\n")

    summary = process_tc_resp_runs(microstate)

    resp_dir = microstate / "terachem" / "respout"
    esp_dir = microstate / "terachem" / "espxyz"

    assert (resp_dir / "conf0001.resp.out").read_text(encoding="utf-8") == "RESP1\n"
    assert (esp_dir / "conf0001.esp.xyz").read_text(encoding="utf-8") == "ESP1\n"
    assert (resp_dir / "conf0002.resp.out").read_text(encoding="utf-8") == "RESP2\n"
    assert (esp_dir / "conf0002.esp.xyz").read_text(encoding="utf-8") == "ESP2\n"

    assert summary["copied_resp"] == 2
    assert summary["copied_esp"] == 2
    assert summary["missing_resp"] == 0
    assert summary["missing_esp"] == 0
