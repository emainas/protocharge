from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

import yaml


def _load_yaml(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping: {path}")
    return data


def _ensure_nowater_files(conf_dir: Path) -> Dict[str, Path]:
    parm7 = conf_dir / "parm7"
    rst7 = conf_dir / "rst7"
    if not parm7.exists():
        cand = list(conf_dir.glob("*.parm7"))
        if cand:
            parm7 = cand[0]
    if not rst7.exists():
        cand = list(conf_dir.glob("*.rst7"))
        if cand:
            rst7 = cand[0]
    if not parm7.exists() or not rst7.exists():
        raise FileNotFoundError(f"Missing parm7/rst7 in {conf_dir}")

    nowater_parm7 = conf_dir / "nowater.parm7"
    nowater_rst7 = conf_dir / "nowater.rst7"
    if not nowater_parm7.exists():
        nowater_parm7.write_bytes(parm7.read_bytes())
    if not nowater_rst7.exists():
        nowater_rst7.write_bytes(rst7.read_bytes())
    return {"parm7": nowater_parm7, "rst7": nowater_rst7}


def _render_resp_in(config: Dict[str, object]) -> str:
    charge = config.get("charge")
    if charge is None:
        raise ValueError("Config must define 'charge'.")
    basis = config.get("basis", "6-31gs")
    method = config.get("method", "rhf")
    spinmult = config.get("spinmult", 1)
    maxit = config.get("maxit", 1000)
    run = config.get("run", "energy")
    resp = config.get("resp", "yes")
    esp_grid_dens = config.get("esp_grid_dens", 4.0)

    lines = [
        "prmtop nowater.parm7",
        "coordinates nowater.rst7",
        f"basis {basis}",
        f"method {method}",
        f"charge {charge}",
        f"spinmult {spinmult}",
        f"maxit {maxit}",
        f"run {run}",
        f"resp {resp}",
        f"esp_grid_dens {esp_grid_dens}",
    ]
    return "\n".join(lines) + "\n"


def _render_slurm(jobname: str) -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH -p l40-gpu",
            f"#SBATCH -J {jobname}",
            "#SBATCH -t 00:05:00",
            "#SBATCH --qos gpu_access",
            "#SBATCH --gres=gpu:1",
            "",
            "ml tc/25.03",
            "terachem resp.in > resp.out",
            "",
        ]
    )


def run_tc_resp(
    microstate_path: Path,
    *,
    submit: bool = True,
) -> List[Dict[str, object]]:
    microstate_path = microstate_path.resolve()
    confs_root = microstate_path / "input_tc_structures" / "confs"
    if not confs_root.is_dir():
        raise FileNotFoundError(f"Missing confs directory: {confs_root}")

    config_root = (
        microstate_path.parent.parent
        / "configs"
        / microstate_path.name
        / "input_tc_structures"
    )

    results: List[Dict[str, object]] = []
    conf_dirs = sorted([p for p in confs_root.iterdir() if p.is_dir()])

    for conf_dir in conf_dirs:
        conf_name = conf_dir.name
        config_path = config_root / conf_name / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config: {config_path}")

        _ensure_nowater_files(conf_dir)
        cfg = _load_yaml(config_path)

        resp_in = conf_dir / "resp.in"
        resp_in.write_text(_render_resp_in(cfg), encoding="utf-8")

        jobname = f"tc_{microstate_path.name}_{conf_name}"
        slurm_path = conf_dir / "run_tc_resp.slurm"
        slurm_path.write_text(_render_slurm(jobname), encoding="utf-8")

        if submit:
            subprocess.run(["sbatch", str(slurm_path)], check=True)

        results.append(
            {
                "conf": conf_name,
                "config": config_path,
                "resp_in": resp_in,
                "slurm": slurm_path,
                "jobname": jobname,
            }
        )

    return results
