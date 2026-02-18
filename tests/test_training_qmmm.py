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


def test_md_prep_generates_tleap(tmp_path: Path) -> None:
    cfg = {
        "microstate": "HID",
        "md": {
            "prep": {
                "workdir": str(tmp_path / "md" / "prep"),
                "tleap": {
                    "module": "ml amber/25",
                    "binary": "tleap",
                    "mol2": str(tmp_path / "ligand.mol2"),
                    "frcmod": str(tmp_path / "ligand.frcmod"),
                    "leaprc_mol": "leaprc.gaff2",
                    "leaprc_sol": "leaprc.water.tip3p",
                    "water_model": "TIP3PBOX",
                    "buffer": 10.0,
                    "prefix": "HID",
                },
            }
        },
    }
    (tmp_path / "ligand.mol2").write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    (tmp_path / "ligand.frcmod").write_text("MASS\n", encoding="utf-8")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(cfg), encoding="utf-8")

    from protocharge.training.md_qmmm import run_md_stage

    run_md_stage("prep", cfg_path, slurm=False, dry_run=True)
    assert (tmp_path / "md" / "prep" / "tleap.in").exists()


def test_md_run_writes_inputs_and_scripts(tmp_path: Path) -> None:
    parm7 = tmp_path / "prep.parm7"
    rst7 = tmp_path / "prep.rst7"
    parm7.write_text("PARM7", encoding="utf-8")
    rst7.write_text("RST7", encoding="utf-8")

    cfg = {
        "microstate": "HID",
        "md": {
            "run": {
                "workdir": str(tmp_path / "md" / "run"),
                "inputs": {"parm7": str(parm7), "rst7": str(rst7)},
                "runtime": {"module": "amber/25", "executable": "pmemd.cuda", "env": {}},
                "stages": [
                    {"name": "min", "description": "Min", "cntrl": {"imin": 1}, "traj": False},
                    {"name": "heat", "description": "Heat", "cntrl": {"imin": 0}, "traj": True},
                ],
                "slurm": {"partition": "batch", "nodes": 1, "cpus_per_task": 4, "time": "01:00:00"},
            }
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(cfg), encoding="utf-8")

    from protocharge.training.md_qmmm import run_md_stage

    run_md_stage("run", cfg_path, slurm=True, dry_run=True)
    workdir = tmp_path / "md" / "run"
    assert (workdir / "min.in").exists()
    assert (workdir / "heat.in").exists()
    assert (workdir / "run.sh").exists()
    assert (workdir / "slurm.sh").exists()
    assert (workdir / "prep.parm7").exists()
    assert (workdir / "prep.rst7").exists()
