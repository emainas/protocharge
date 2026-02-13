from __future__ import annotations

from pathlib import Path

from biliresp.terachem_processing import process_terachem_outputs


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_process_terachem_outputs(tmp_path: Path) -> None:
    microstate = tmp_path / "HID"
    raw_dir = microstate / "raw_terachem_outputs"

    # frame1 with two configs; frame2 with one config missing esp.xyz
    _write(
        raw_dir / "frame1" / "scr.carved_frame1" / "splits" / "conf1" / "resp-vac.out",
        "RESP1\n",
    )
    _write(
        raw_dir
        / "frame1"
        / "scr.carved_frame1"
        / "splits"
        / "conf1"
        / "scr.nowater"
        / "esp.xyz",
        "ESP1\n",
    )
    _write(
        raw_dir / "frame1" / "scr.carved_frame1" / "splits" / "conf2" / "resp-vac.out",
        "RESP2\n",
    )
    _write(
        raw_dir
        / "frame1"
        / "scr.carved_frame1"
        / "splits"
        / "conf2"
        / "scr.nowater"
        / "esp.xyz",
        "ESP2\n",
    )
    _write(
        raw_dir / "frame2" / "scr.carved_frame2" / "splits" / "conf1" / "resp-vac.out",
        "RESP3\n",
    )

    summary = process_terachem_outputs(microstate)

    resp_dir = microstate / "terachem" / "respout"
    esp_dir = microstate / "terachem" / "espxyz"

    assert (resp_dir / "conf0001.resp.out").read_text(encoding="utf-8") == "RESP1\n"
    assert (esp_dir / "conf0001.esp.xyz").read_text(encoding="utf-8") == "ESP1\n"
    assert (resp_dir / "conf0002.resp.out").read_text(encoding="utf-8") == "RESP2\n"
    assert (esp_dir / "conf0002.esp.xyz").read_text(encoding="utf-8") == "ESP2\n"
    assert (resp_dir / "conf0003.resp.out").read_text(encoding="utf-8") == "RESP3\n"

    assert summary["frames"] == 2
    assert summary["confs"] == 3
    assert summary["copied_resp"] == 3
    assert summary["copied_esp"] == 2
    assert summary["missing_resp"] == 0
    assert summary["missing_esp"] == 1
