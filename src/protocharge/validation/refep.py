from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import yaml
import numpy as np

from protocharge.paths import results_root


def _load_yaml(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping: {path}")
    return data


def _resolve_path(value: object, base_dir: Path) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Expected a path string, got {type(value).__name__}")
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _normalize_commands(commands: Sequence[object]) -> List[List[str]]:
    normalized: List[List[str]] = []
    for cmd in commands:
        if isinstance(cmd, str):
            normalized.append(cmd.split())
        elif isinstance(cmd, (list, tuple)):
            normalized.append([str(c) for c in cmd])
        else:
            raise ValueError(f"Unsupported command type: {type(cmd).__name__}")
    return normalized


def _load_charge_overrides(path: Path) -> List[Dict[str, object]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = []
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            if "." not in key:
                continue
            resid, atom = key.split(".", 1)
            items.append({"resid": resid, "atom": atom, "charge": value})
        return items
    raise ValueError(f"Charge overrides must be a list or mapping: {path}")


def _render_charge_block(overrides: Iterable[Mapping[str, object]]) -> str:
    lines = []
    for entry in overrides:
        resid = entry.get("resid")
        atom = entry.get("atom")
        charge = entry.get("charge")
        if resid is None or atom is None or charge is None:
            continue
        lines.append(f"set {resid}.{atom} charge {charge}")
    return "\n".join(lines) + ("\n" if lines else "")


def _insert_charge_block(tleap_text: str, block: str, marker: str) -> str:
    if not block:
        return tleap_text
    if marker in tleap_text:
        return tleap_text.replace(marker, marker + "\n" + block)
    return tleap_text.rstrip() + "\n\n# BILIRESP_CHARGES\n" + block


def _parse_mol2_atoms(path: Path) -> List[Dict[str, object]]:
    atoms: List[Dict[str, object]] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if not in_atoms or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        atom_name = parts[1]
        subst_id = parts[6]
        subst_name = parts[7]
        atoms.append({"atom": atom_name, "subst_id": subst_id, "subst_name": subst_name})
    if not atoms:
        raise ValueError(f"No atoms parsed from mol2: {path}")
    return atoms


def _select_charges(
    arr: np.ndarray,
    frame: int | None,
    aggregate: str | None,
    frames_axis: int,
) -> np.ndarray:
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim != 2:
        raise ValueError(f"Charges array must be 1D or 2D, got shape {arr.shape}")
    if frames_axis not in (0, 1):
        raise ValueError("frames_axis must be 0 or 1")
    if frames_axis == 0:
        arr = arr.T
    if aggregate:
        if aggregate == "mean":
            return arr.mean(axis=1)
        if aggregate == "median":
            return np.median(arr, axis=1)
        raise ValueError(f"Unsupported aggregate: {aggregate}")
    if frame is None:
        frame = -1
    return arr[:, frame]


def _load_charges(path: Path, key: str | None, frame: int | None, aggregate: str | None, frames_axis: int) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return _select_charges(arr, frame, aggregate, frames_axis)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if key is None:
            for cand in ("step2", "charges", "q", "charges_final", "step1"):
                if cand in data:
                    key = cand
                    break
        if key is None or key not in data:
            raise ValueError(f"Missing charge key in {path}; specify charges.key")
        return _select_charges(data[key], frame, aggregate, frames_axis)
    raise ValueError(f"Unsupported charge file type: {path}")


def _export_charge_yaml(
    charges: np.ndarray,
    mol2_path: Path,
    output_path: Path,
    resid_prefix: str | None,
    resid_map: Mapping[str, str] | None,
    resid_source: str,
) -> Path:
    atoms = _parse_mol2_atoms(mol2_path)
    if len(charges) != len(atoms):
        raise ValueError(
            f"Charge count {len(charges)} does not match mol2 atoms {len(atoms)}"
        )

    entries: List[Dict[str, object]] = []
    for atom, charge in zip(atoms, charges):
        resid = None
        if resid_map:
            key = str(atom.get(resid_source))
            resid = resid_map.get(key)
        if resid is None:
            resid_val = str(atom.get(resid_source))
            if resid_prefix:
                resid = f"{resid_prefix}.{resid_val}"
            else:
                resid = resid_val
        entries.append({"resid": resid, "atom": atom["atom"], "charge": float(charge)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return output_path


def _resolve_slurm_config(slurm: Mapping[str, object]) -> Mapping[str, object]:
    profiles = slurm.get("profiles")
    profile_name = slurm.get("profile")
    if profile_name and isinstance(profiles, dict):
        profile = profiles.get(profile_name)
        if isinstance(profile, dict):
            merged = dict(profile)
            for key, value in slurm.items():
                if key in {"profiles", "profile"}:
                    continue
                merged[key] = value
            return merged
    return slurm


def _render_slurm(job_name: str, commands: List[List[str]], slurm: Mapping[str, object]) -> str:
    slurm_cfg = _resolve_slurm_config(slurm)
    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_cfg.get('job_name', job_name)}",
        f"#SBATCH --output={slurm_cfg.get('output', 'slurm_logs/%x.%j.out')}",
        f"#SBATCH --error={slurm_cfg.get('error', 'slurm_logs/%x.%j.err')}",
        f"#SBATCH --time={slurm_cfg.get('time', '24:00:00')}",
        f"#SBATCH --nodes={slurm_cfg.get('nodes', 1)}",
        f"#SBATCH --cpus-per-task={slurm_cfg.get('cpus_per_task', 12)}",
        f"#SBATCH --mem={slurm_cfg.get('mem', '64G')}",
    ]
    if "partition" in slurm_cfg:
        header.append(f"#SBATCH --partition={slurm_cfg['partition']}")
    if "account" in slurm_cfg:
        header.append(f"#SBATCH --account={slurm_cfg['account']}")
    if "qos" in slurm_cfg:
        header.append(f"#SBATCH --qos={slurm_cfg['qos']}")
    if "gres" in slurm_cfg:
        header.append(f"#SBATCH --gres={slurm_cfg['gres']}")
    if "constraint" in slurm_cfg:
        header.append(f"#SBATCH --constraint={slurm_cfg['constraint']}")
    extra = slurm_cfg.get("extra")
    if isinstance(extra, list):
        for line in extra:
            header.append(str(line))
    lines = header + [
        "",
        "set -euo pipefail",
        "set -x",
        "",
        "module load amber/24 || true",
        "",
    ]
    for cmd in commands:
        lines.append(" ".join(cmd))
    return "\n".join(lines) + "\n"


def run_refep_stage(
    stage: str,
    config_path: Path,
    *,
    slurm: bool = False,
    dry_run: bool = False,
) -> Path:
    cfg = _load_yaml(config_path)
    microstate = cfg.get("microstate")
    if not microstate:
        raise ValueError("Config must define 'microstate'.")

    refep = cfg.get("refep", {})
    if not isinstance(refep, dict):
        raise ValueError("Config 'refep' must be a mapping.")

    stage_cfg = refep.get(stage)
    if not isinstance(stage_cfg, dict):
        raise ValueError(f"Config missing refep.{stage} mapping.")

    base_dir = Path(stage_cfg.get("workdir") or (results_root() / microstate / "refep" / stage))
    _ensure_dir(base_dir)

    # Charge exports (prep stage)
    if stage == "prep":
        exports = stage_cfg.get("charge_exports", [])
        if isinstance(exports, list):
            for entry in exports:
                if not isinstance(entry, dict):
                    continue
                charges_cfg = entry.get("charges", {})
                if not isinstance(charges_cfg, dict):
                    raise ValueError("charge_exports.charges must be a mapping")
                charges_path = _resolve_path(charges_cfg.get("path"), config_path.parent)
                key = charges_cfg.get("key")
                frame = charges_cfg.get("frame")
                aggregate = charges_cfg.get("aggregate")
                frames_axis = int(charges_cfg.get("frames_axis", 1))

                mol2_path = _resolve_path(entry.get("mol2"), config_path.parent)
                output_path = _resolve_path(entry.get("output"), config_path.parent)
                resid_prefix = entry.get("resid_prefix")
                resid_map = entry.get("resid_map") if isinstance(entry.get("resid_map"), dict) else None
                resid_source = entry.get("resid_source", "subst_id")

                charges = _load_charges(charges_path, key, frame, aggregate, frames_axis)
                _export_charge_yaml(
                    charges,
                    mol2_path,
                    output_path,
                    resid_prefix,
                    resid_map,
                    resid_source,
                )

    # Copy declared files
    files = stage_cfg.get("files", [])
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            src = _resolve_path(item.get("src"), config_path.parent)
            dst = base_dir / str(item.get("dst") or src.name)
            _copy_file(src, dst)

    # Optional tleap helper with charge overrides (prep stage)
    if stage == "prep":
        tleap_inputs = stage_cfg.get("tleap_inputs", [])
        if isinstance(tleap_inputs, list):
            for entry in tleap_inputs:
                if not isinstance(entry, dict):
                    continue
                template = _resolve_path(entry.get("template"), config_path.parent)
                out_name = entry.get("output", template.name)
                out_path = base_dir / str(out_name)
                text = template.read_text(encoding="utf-8")
                charges_path = entry.get("charges")
                marker = entry.get("marker", "# BILIRESP_CHARGES")
                if charges_path:
                    overrides = _load_charge_overrides(_resolve_path(charges_path, config_path.parent))
                    block = _render_charge_block(overrides)
                    text = _insert_charge_block(text, block, marker)
                out_path.write_text(text, encoding="utf-8")

    commands = _normalize_commands(stage_cfg.get("commands", []))

    if slurm and commands:
        slurm_cfg = stage_cfg.get("slurm", {})
        script = _render_slurm(f"refep_{stage}_{microstate}", commands, slurm_cfg)
        script_path = base_dir / f"refep_{stage}.slurm"
        script_path.write_text(script, encoding="utf-8")
        if not dry_run:
            subprocess.run(["sbatch", str(script_path)], check=True)
        return base_dir

    if commands and not dry_run:
        for cmd in commands:
            subprocess.run(cmd, check=True, cwd=base_dir)

    return base_dir
