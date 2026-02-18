from __future__ import annotations

from pathlib import Path

from protocharge.generator.run_tc_resp import run_tc_resp


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_run_tc_resp_writes_inputs(tmp_path: Path) -> None:
    microstate = tmp_path / "input" / "microstates" / "HID"
    conf_root = microstate / "input_tc_structures" / "confs"
    config_root = tmp_path / "input" / "configs" / "HID" / "input_tc_structures"

    _write(conf_root / "conf1" / "parm7", "PARM7\n")
    _write(conf_root / "conf1" / "rst7", "RST7\n")
    _write(conf_root / "conf2" / "parm7", "PARM7-2\n")
    _write(conf_root / "conf2" / "rst7", "RST7-2\n")

    _write(
        config_root / "conf1" / "config.yaml",
        "charge: 1\nbasis: 6-31gs\nmethod: rhf\n",
    )
    _write(
        config_root / "conf2" / "config.yaml",
        "charge: 0\nbasis: 6-31gs\nmethod: rhf\n",
    )

    results = run_tc_resp(microstate, submit=False)

    assert len(results) == 2
    for conf in ("conf1", "conf2"):
        conf_dir = conf_root / conf
        assert (conf_dir / "nowater.parm7").exists()
        assert (conf_dir / "nowater.rst7").exists()
        assert (conf_dir / "resp.in").exists()
        assert (conf_dir / "run_tc_resp.slurm").exists()
