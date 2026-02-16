from __future__ import annotations

from pathlib import Path

from protocharge.training.md_qmmm import run_qmmm_stage


def test_qmmm_prep_writes_region_and_inputs(tmp_path: Path) -> None:
    cfg = {
        "microstate": "HID",
        "qmmm": {
            "prep": {
                "workdir": str(tmp_path / "qmmm"),
                "parm7": str(tmp_path / "dummy.parm7"),
                "traj": str(tmp_path / "dummy.nc"),
                "frames": {"start": 1, "end": 2, "step": 1},
                "ambermask": "339",
                "closest_n": 50,
                "region_range": "0-3",
                "charge": 0,
                "tc": {},
            }
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(cfg), encoding="utf-8")

    run_qmmm_stage("prep", cfg_path, slurm=False, dry_run=True)

    region = (tmp_path / "qmmm" / "region.qm").read_text(encoding="utf-8")
    assert region.strip().splitlines() == ["0", "1", "2", "3"]
    tc_in = tmp_path / "qmmm" / "frames" / "frame1" / "tc.in"
    assert tc_in.exists()
