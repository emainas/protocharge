from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from protocharge.validation import refep as refep_utils
from protocharge.paths import project_root


def _load_yaml(path: Path) -> Dict[str, object]:
    return refep_utils._load_yaml(path)


def _resolve_path(value: object, base_dir: Path) -> Path:
    return refep_utils._resolve_path(value, base_dir)


def _resolve_path_with_base(value: object, config_dir: Path) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Expected a path string, got {type(value).__name__}")
    path = Path(value)
    if path.is_absolute():
        return path
    cand = config_dir / path
    if cand.exists():
        return cand
    return project_root() / path


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


def _render_mdin(stage: Mapping[str, object]) -> str:
    description = str(stage.get("description", "MD stage"))
    cntrl = stage.get("cntrl")
    if not isinstance(cntrl, dict):
        raise ValueError("md.run.stages[*].cntrl must be a mapping.")
    lines = [description, "&cntrl"]
    for key, value in cntrl.items():
        lines.append(f"  {key}={value},")
    lines.append("/")

    wt = stage.get("wt")
    if isinstance(wt, list):
        for card in wt:
            if not isinstance(card, dict):
                continue
            card_type = str(card.get("type", "END"))
            if card_type.upper() == "END":
                lines.append("&wt type='END' /")
                continue
            parts = [f"type='{card_type}'"]
            for key in ("istep1", "istep2", "value1", "value2"):
                if key in card:
                    parts.append(f"{key}={card[key]}")
            lines.append("&wt " + ", ".join(parts) + " /")

    return "\n".join(lines) + "\n"


def _write_md_run_scripts(
    base_dir: Path,
    parm7_name: str,
    rst7_name: str,
    runtime: Mapping[str, object],
    stages: List[Mapping[str, object]],
) -> Path:
    strict = runtime.get("strict_mode", True)
    module_name = runtime.get("module")
    executable = runtime.get("executable", "pmemd.cuda")
    env = runtime.get("env", {})

    strict_line = "set -euo pipefail" if strict else ""
    module_block = f"module load {module_name}" if module_name else ""
    env_block = "\n".join([f"export {k}={v}" for k, v in env.items()])

    lines = [
        "#!/usr/bin/env bash",
        strict_line,
        "",
        "module purge",
        module_block,
        "",
        env_block,
        "",
        f'echo "==> Running MD stages with {executable}"',
        "",
    ]

    current_rst7 = rst7_name
    for stage in stages:
        name = stage["name"]
        out_rst7 = f"{name}.rst7"
        out_nc = f"{name}.nc"
        out_info = f"{name}.info"
        out_out = f"{name}.out"
        traj = stage.get("traj")
        if traj is None:
            traj = not str(name).startswith("min")

        lines.append(f'if [[ ! -f {out_out} ]]; then')
        lines.append(f'  echo "  {name}..."')
        lines.append(f"  {executable} -O \\")
        lines.append(f"    -i {name}.in \\")
        lines.append(f"    -p {parm7_name} \\")
        lines.append(f"    -c {current_rst7} \\")
        lines.append(f"    -r {out_rst7} \\")
        lines.append(f"    -o {out_out} \\")
        lines.append(f"    -inf {out_info} \\")
        if traj:
            lines.append(f"    -x {out_nc}")
        else:
            lines[-1] = lines[-1].rstrip(" \\")
        lines.append("else")
        lines.append(f'  echo "  Skipping {name} ({out_out} exists)"')
        lines.append("fi")
        lines.append("")
        current_rst7 = out_rst7

    script_path = base_dir / "run.sh"
    script_path.write_text("\n".join([l for l in lines if l is not None]) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _write_md_slurm(base_dir: Path, slurm_cfg: Mapping[str, object]) -> Path:
    cfg = refep_utils._resolve_slurm_config(slurm_cfg)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {cfg.get('job_name', 'md-run')}",
        f"#SBATCH -p {cfg.get('partition', 'batch')}",
        f"#SBATCH -N {cfg.get('nodes', 1)}",
        f"#SBATCH -t {cfg.get('time', '24:00:00')}",
        f"#SBATCH --cpus-per-task={cfg.get('cpus_per_task', 16)}",
    ]
    if "output" in cfg:
        lines.append(f"#SBATCH -o {cfg['output']}")
    if "error" in cfg:
        lines.append(f"#SBATCH -e {cfg['error']}")
    if "qos" in cfg:
        lines.append(f"#SBATCH --qos={cfg['qos']}")
    if "gres" in cfg:
        lines.append(f"#SBATCH --gres={cfg['gres']}")
    if "account" in cfg:
        lines.append(f"#SBATCH --account={cfg['account']}")
    if "mem" in cfg:
        lines.append(f"#SBATCH --mem={cfg['mem']}")
    extra = cfg.get("extra")
    if isinstance(extra, list):
        for line in extra:
            lines.append(str(line))
    lines += [
        "",
        "bash run.sh",
        "",
    ]
    script_path = base_dir / "slurm.sh"
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path

def _run_generic_stage(stage_cfg: Mapping[str, object], base_dir: Path, config_dir: Path, slurm: bool, dry_run: bool) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    exports = stage_cfg.get("charge_exports", [])
    if isinstance(exports, list):
        for entry in exports:
            if not isinstance(entry, dict):
                continue
            charges_cfg = entry.get("charges", {})
            if not isinstance(charges_cfg, dict):
                raise ValueError("charge_exports.charges must be a mapping")
            charges_path = _resolve_path_with_base(charges_cfg.get("path"), config_dir)
            key = charges_cfg.get("key")
            frame = charges_cfg.get("frame")
            aggregate = charges_cfg.get("aggregate")
            frames_axis = int(charges_cfg.get("frames_axis", 1))

            mol2_path = _resolve_path_with_base(entry.get("mol2"), config_dir)
            output_path = _resolve_path_with_base(entry.get("output"), config_dir)
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
            src = _resolve_path_with_base(item.get("src"), config_dir)
            dst = base_dir / str(item.get("dst") or src.name)
            refep_utils._copy_file(src, dst)

    tleap_cfg = stage_cfg.get("tleap")
    if isinstance(tleap_cfg, dict):
        tleap_in = base_dir / "tleap.in"
        mol2 = _resolve_path_with_base(tleap_cfg.get("mol2"), config_dir)
        frcmod = _resolve_path_with_base(tleap_cfg.get("frcmod"), config_dir)
        frcmod_ion = tleap_cfg.get("frcmod_ion")
        leaprc_mol = tleap_cfg.get("leaprc_mol", "leaprc.gaff2")
        leaprc_sol = tleap_cfg.get("leaprc_sol", "leaprc.water.tip3p")
        water_model = tleap_cfg.get("water_model", "TIP3PBOX")
        buffer = float(tleap_cfg.get("buffer", 15.0))
        counterion = tleap_cfg.get("counterion")
        counterion_num = int(tleap_cfg.get("counterion_num", 0))
        prefix = tleap_cfg.get("prefix", "solv")

        base_dir.mkdir(parents=True, exist_ok=True)
        if mol2.exists():
            refep_utils._copy_file(mol2, base_dir / mol2.name)
        if frcmod.exists():
            refep_utils._copy_file(frcmod, base_dir / frcmod.name)

        addions_block = ""
        if counterion and counterion_num > 0:
            addions_block = f"addions sys {counterion} {counterion_num}\n"

        ion_block = ""
        if frcmod_ion:
            ion_block = f"loadamberparams {frcmod_ion}\n"

        tleap_text = (
            "# Auto-generated by protocharge\n"
            f"source {leaprc_mol}\n"
            f"source {leaprc_sol}\n"
            f"{ion_block}"
            f"loadamberparams {frcmod.name}\n"
            f"sys = loadmol2 {mol2.name}\n"
            f"solvatebox sys {water_model} {buffer}\n"
            f"{addions_block}"
            f"saveamberparm sys {prefix}.parm7 {prefix}.rst7\n"
            "quit\n"
        )
        tleap_in.write_text(tleap_text, encoding="utf-8")

        module_name = tleap_cfg.get("module")
        tleap_bin = tleap_cfg.get("binary", "tleap")
        if module_name and not dry_run:
            subprocess.run(
                ["bash", "-lc", f"ml {module_name} && {tleap_bin} -f {tleap_in.name}"],
                check=True,
                cwd=base_dir,
            )

    tleap_inputs = stage_cfg.get("tleap_inputs", [])
    if isinstance(tleap_inputs, list):
        for entry in tleap_inputs:
            if not isinstance(entry, dict):
                continue
            template = _resolve_path_with_base(entry.get("template"), config_dir)
            out_name = entry.get("output", template.name)
            out_path = base_dir / str(out_name)
            text = template.read_text(encoding="utf-8")
            charges_path = entry.get("charges")
            marker = entry.get("marker", "# BILIRESP_CHARGES")
            if charges_path:
                overrides = refep_utils._load_charge_overrides(_resolve_path_with_base(charges_path, config_dir))
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

    if stage == "run":
        base_dir.mkdir(parents=True, exist_ok=True)
        inputs = stage_cfg.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError("md.run.inputs must be a mapping.")
        parm7_path = _resolve_path_with_base(inputs.get("parm7"), config_path.parent)
        rst7_path = _resolve_path_with_base(inputs.get("rst7"), config_path.parent)
        if not parm7_path.exists() or not rst7_path.exists():
            raise FileNotFoundError("md.run.inputs parm7/rst7 not found.")

        parm7_name = parm7_path.name
        rst7_name = rst7_path.name
        refep_utils._copy_file(parm7_path, base_dir / parm7_name)
        refep_utils._copy_file(rst7_path, base_dir / rst7_name)

        stages = stage_cfg.get("stages", [])
        if not isinstance(stages, list) or not stages:
            raise ValueError("md.run.stages must be a non-empty list.")
        stage_defs: List[Mapping[str, object]] = []
        for entry in stages:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                raise ValueError("md.run.stages entries require a 'name'.")
            stage_defs.append(entry)
            mdin = base_dir / f"{name}.in"
            mdin.write_text(_render_mdin(entry), encoding="utf-8")

        runtime = stage_cfg.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("md.run.runtime must be a mapping.")
        run_sh = _write_md_run_scripts(base_dir, parm7_name, rst7_name, runtime, stage_defs)

        slurm_sh = None
        slurm_cfg = stage_cfg.get("slurm", {})
        if slurm and isinstance(slurm_cfg, dict):
            slurm_sh = _write_md_slurm(base_dir, slurm_cfg)

        (base_dir / "spec.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

        if slurm and slurm_sh and not dry_run:
            subprocess.run(["sbatch", slurm_sh.name], check=True, cwd=base_dir)
        elif not slurm and not dry_run:
            subprocess.run(["bash", run_sh.name], check=True, cwd=base_dir)
        return base_dir

    _run_generic_stage(stage_cfg, base_dir, config_path.parent, slurm, dry_run)
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
