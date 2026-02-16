from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from biliresp.validation import refep as refep_utils


def _load_yaml(path: Path) -> Dict[str, object]:
    return refep_utils._load_yaml(path)


def _resolve_path(value: object, base_dir: Path) -> Path:
    return refep_utils._resolve_path(value, base_dir)


def _normalize_commands(commands: Sequence[object]) -> List[List[str]]:
    return refep_utils._normalize_commands(commands)


def _render_slurm(job_name: str, commands: List[List[str]], slurm: Mapping[str, object]) -> str:
    return refep_utils._render_slurm(job_name, commands, slurm)


def _write_region_qm(range_spec: str, output_path: Path) -> None:
    entries: List[int] = []
    for chunk in range_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            entries.extend(range(int(start), int(end) + 1))
        else:
            entries.append(int(chunk))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(i) for i in entries) + "\n", encoding="utf-8")


def _render_tc_in(tc_cfg: Mapping[str, object], charge: int | float, region_qm: str | None) -> str:
    basis = tc_cfg.get("basis", "def2-svp")
    method = tc_cfg.get("method", "wb97xd3")
    dftd = tc_cfg.get("dftd", "d3")
    spinmult = tc_cfg.get("spinmult", 1)
    maxit = tc_cfg.get("maxit", 1000)
    run = tc_cfg.get("run", "md")
    nstep = tc_cfg.get("nstep", 20000)
    timestep = tc_cfg.get("timestep", 0.5)
    mdbc = tc_cfg.get("mdbc", "spherical")
    wmodel = tc_cfg.get("wmodel", "tip3p")

    lines = [
        "prmtop nobox.parm7",
        "coordinates carved_frame.rst7",
    ]
    if region_qm:
        lines.append("qmindices region.qm")
    lines += [
        f"basis {basis}",
        f"method {method}",
        f"charge {charge}",
        f"dftd {dftd}",
        f"spinmult {spinmult}",
        f"maxit {maxit}",
        f"run {run}",
        f"nstep {nstep}",
        f"timestep {timestep}",
        f"mdbc {mdbc}",
        f"wmodel {wmodel}",
    ]
    return "\n".join(lines) + "\n"


def _render_tc_resp_in(tc_cfg: Mapping[str, object], charge: int | float, region_qm: str | None) -> str:
    basis = tc_cfg.get("basis", "def2-svp")
    method = tc_cfg.get("method", "wb97xd3")
    dftd = tc_cfg.get("dftd", "d3")
    spinmult = tc_cfg.get("spinmult", 1)
    maxit = tc_cfg.get("maxit", 1000)
    esp_grid_dens = tc_cfg.get("esp_grid_dens", 4.0)

    lines = [
        "prmtop nobox.parm7",
        "coordinates coords.rst7",
    ]
    if region_qm:
        lines.append("qmindices region.qm")
    lines += [
        f"basis {basis}",
        f"method {method}",
        f"charge {charge}",
        f"dftd {dftd}",
        f"spinmult {spinmult}",
        f"maxit {maxit}",
        "run energy",
        "resp yes",
        f"esp_grid_dens {esp_grid_dens}",
    ]
    return "\n".join(lines) + "\n"


def _render_tc_slurm(jobname: str) -> str:
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


def _run_generic_stage(stage_cfg: Mapping[str, object], base_dir: Path, slurm: bool, dry_run: bool) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    exports = stage_cfg.get("charge_exports", [])
    if isinstance(exports, list):
        for entry in exports:
            if not isinstance(entry, dict):
                continue
            charges_cfg = entry.get("charges", {})
            if not isinstance(charges_cfg, dict):
                raise ValueError("charge_exports.charges must be a mapping")
            charges_path = _resolve_path(charges_cfg.get("path"), base_dir)
            key = charges_cfg.get("key")
            frame = charges_cfg.get("frame")
            aggregate = charges_cfg.get("aggregate")
            frames_axis = int(charges_cfg.get("frames_axis", 1))

            mol2_path = _resolve_path(entry.get("mol2"), base_dir)
            output_path = _resolve_path(entry.get("output"), base_dir)
            resid_prefix = entry.get("resid_prefix")
            resid_map = entry.get("resid_map") if isinstance(entry.get("resid_map"), dict) else None
            resid_source = entry.get("resid_source", "subst_id")

            charges = refep_utils._load_charges(charges_path, key, frame, aggregate, frames_axis)
            refep_utils._export_charge_yaml(
                charges,
                mol2_path,
                output_path,
                resid_prefix,
                resid_map,
                resid_source,
            )

    files = stage_cfg.get("files", [])
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            src = _resolve_path(item.get("src"), base_dir)
            dst = base_dir / str(item.get("dst") or src.name)
            refep_utils._copy_file(src, dst)

    tleap_inputs = stage_cfg.get("tleap_inputs", [])
    if isinstance(tleap_inputs, list):
        for entry in tleap_inputs:
            if not isinstance(entry, dict):
                continue
            template = _resolve_path(entry.get("template"), base_dir)
            out_name = entry.get("output", template.name)
            out_path = base_dir / str(out_name)
            text = template.read_text(encoding="utf-8")
            charges_path = entry.get("charges")
            marker = entry.get("marker", "# BILIRESP_CHARGES")
            if charges_path:
                overrides = refep_utils._load_charge_overrides(_resolve_path(charges_path, base_dir))
                block = refep_utils._render_charge_block(overrides)
                text = refep_utils._insert_charge_block(text, block, marker)
            out_path.write_text(text, encoding="utf-8")

    commands = _normalize_commands(stage_cfg.get("commands", []))
    if slurm and commands:
        slurm_cfg = stage_cfg.get("slurm", {})
        script = _render_slurm(stage_cfg.get("job_name", "mdqmmm"), commands, slurm_cfg)
        script_path = base_dir / "stage.slurm"
        script_path.write_text(script, encoding="utf-8")
        if not dry_run:
            subprocess.run(["sbatch", str(script_path)], check=True)
        return

    if commands and not dry_run:
        for cmd in commands:
            subprocess.run(cmd, check=True, cwd=base_dir)


def run_md_stage(stage: str, config_path: Path, *, slurm: bool, dry_run: bool) -> Path:
    cfg = _load_yaml(config_path)
    md = cfg.get("md", {})
    if not isinstance(md, dict):
        raise ValueError("Config 'md' must be a mapping.")
    stage_cfg = md.get(stage)
    if not isinstance(stage_cfg, dict):
        raise ValueError(f"Config missing md.{stage} mapping.")
    base_dir = Path(stage_cfg.get("workdir"))
    _run_generic_stage(stage_cfg, base_dir, slurm, dry_run)
    return base_dir


def run_qmmm_stage(stage: str, config_path: Path, *, slurm: bool, dry_run: bool) -> Path:
    cfg = _load_yaml(config_path)
    qmmm = cfg.get("qmmm", {})
    if not isinstance(qmmm, dict):
        raise ValueError("Config 'qmmm' must be a mapping.")
    stage_cfg = qmmm.get(stage)
    if not isinstance(stage_cfg, dict):
        raise ValueError(f"Config missing qmmm.{stage} mapping.")

    base_dir = Path(stage_cfg.get("workdir"))
    base_dir.mkdir(parents=True, exist_ok=True)

    if stage == "prep":
        parm7 = _resolve_path(stage_cfg.get("parm7"), config_path.parent)
        traj = _resolve_path(stage_cfg.get("traj"), config_path.parent)
        frames = stage_cfg.get("frames", {})
        start = int(frames.get("start"))
        end = int(frames.get("end"))
        step = int(frames.get("step"))
        ambermask = str(stage_cfg.get("ambermask"))
        closest_n = int(stage_cfg.get("closest_n"))
        region_range = str(stage_cfg.get("region_range"))
        charge = stage_cfg.get("charge", 0)
        tc_cfg = stage_cfg.get("tc", {})

        region_qm_path = base_dir / "region.qm"
        _write_region_qm(region_range, region_qm_path)

        frame_dir = base_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        counter = 1
        for frame in range(start, end + 1, step):
            work = frame_dir / f"frame{counter}"
            work.mkdir(parents=True, exist_ok=True)

            cpp_frame = work / "cpptraj.frame.in"
            cpp_carve = work / "cpptraj.carve.in"
            parmed_in = work / "parmed.in"

            cpp_frame.write_text(
                f"parm {parm7}\ntrajin {traj} {frame} {frame}\ntrajout frame.rst7 rst7\nrun\n",
                encoding="utf-8",
            )
            cpp_carve.write_text(
                "\n".join(
                    [
                        f"parm {parm7}",
                        "trajin frame.rst7",
                        f"strip @Cl-",
                        f"closest {closest_n} :{ambermask} outprefix carved_{counter} center",
                        "trajout carved_frame.rst7",
                        "run",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parmed_in.write_text(
                "\n".join(
                    [
                        "parm carved_topology.parm7",
                        "strip :Cl- nobox",
                        "outparm nobox.parm7",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            tc_in = work / "tc.in"
            tc_in.write_text(_render_tc_in(tc_cfg, charge, "region.qm"), encoding="utf-8")
            run_sh = work / "run.sh"
            run_sh.write_text(_render_tc_slurm(f"{counter}.{cfg.get('microstate','state').lower()}"), encoding="utf-8")
            run_sh.chmod(0o755)

            if not dry_run:
                subprocess.run(["cpptraj", "-i", str(cpp_frame)], check=True, cwd=work)
                subprocess.run(["cpptraj", "-i", str(cpp_carve)], check=True, cwd=work)
                carved_candidates = list(work.glob(f"carved_{counter}*.parm7"))
                if not carved_candidates:
                    raise FileNotFoundError(f"Missing carved topology in {work}")
                (work / "carved_topology.parm7").write_bytes(carved_candidates[0].read_bytes())
                subprocess.run(["parmed", "-i", str(parmed_in)], check=True, cwd=work)

            (work / "region.qm").write_text(region_qm_path.read_text(encoding="utf-8"), encoding="utf-8")

            counter += 1
        return base_dir

    if stage == "run":
        if not slurm:
            raise ValueError("qmmm.run expects --slurm to submit jobs.")
        frame_dir = base_dir / "frames"
        frames = sorted([p for p in frame_dir.iterdir() if p.is_dir()])
        for frame in frames:
            run_sh = frame / "run.sh"
            if run_sh.exists() and not dry_run:
                subprocess.run(["sbatch", str(run_sh)], check=True)
        return base_dir

    if stage == "esp":
        frames_cfg = stage_cfg.get("frames", {})
        start = int(frames_cfg.get("start"))
        end = int(frames_cfg.get("end"))
        step = int(frames_cfg.get("step"))
        charge = stage_cfg.get("charge", 0)
        tc_cfg = stage_cfg.get("tc", {})

        frame_dir = base_dir / "frames"
        frames = sorted([p for p in frame_dir.iterdir() if p.is_dir()])
        for frame in frames:
            frame_id = frame.name.replace("frame", "")
            scr_dir = frame / f"scr.carved_frame{frame_id}"
            traj = scr_dir / "coors.dcd"
            if not traj.exists():
                continue
            splits = scr_dir / "splits"
            splits.mkdir(parents=True, exist_ok=True)

            counter = 1
            for step_idx in range(start, end + 1, step):
                conf_dir = splits / f"{counter}"
                conf_dir.mkdir(parents=True, exist_ok=True)
                cpptraj_in = conf_dir / "cpptraj.in"
                cpptraj_in.write_text(
                    "\n".join(
                        [
                            f"parm {frame / 'nobox.parm7'}",
                            f"trajin {traj} {step_idx} {step_idx}",
                            "trajout coords.rst7 rst7",
                            "run",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if not dry_run:
                    subprocess.run(["cpptraj", "-i", str(cpptraj_in)], check=True, cwd=conf_dir)

                (conf_dir / "region.qm").write_text((frame / "region.qm").read_text(encoding="utf-8"), encoding="utf-8")
                resp_in = conf_dir / "resp.in"
                resp_in.write_text(_render_tc_resp_in(tc_cfg, charge, "region.qm"), encoding="utf-8")
                run_sh = conf_dir / "run.sh"
                run_sh.write_text(_render_tc_slurm(f"{frame_id}.{cfg.get('microstate','state').lower()}.{step_idx}"), encoding="utf-8")
                run_sh.chmod(0o755)
                if slurm and not dry_run:
                    subprocess.run(["sbatch", str(run_sh)], check=True)
                counter += 1
        return base_dir

    raise ValueError(f"Unknown qmmm stage: {stage}")
