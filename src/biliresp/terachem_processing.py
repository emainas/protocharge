from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def _natural_key(name: str) -> Tuple[int | str, ...]:
    parts: List[int | str] = []
    buf = ""
    for ch in name:
        if ch.isdigit():
            buf += ch
        else:
            if buf:
                parts.append(int(buf))
                buf = ""
            parts.append(ch)
    if buf:
        parts.append(int(buf))
    return tuple(parts)


def _find_frame_dirs(tc_raw_dir: Path) -> List[Path]:
    frames = []
    for frame in tc_raw_dir.glob("frame*"):
        if not frame.is_dir():
            continue
        inner = frame / f"scr.carved_{frame.name}" / "splits"
        if inner.is_dir():
            frames.append(frame)
    return sorted(frames, key=lambda p: _natural_key(p.name))


def process_terachem_outputs(
    microstate_path: Path,
    tc_raw_subdir: str = "raw_terachem_outputs",
) -> Dict[str, object]:
    microstate_path = microstate_path.resolve()
    tc_raw_dir = microstate_path / tc_raw_subdir
    if microstate_path.name == tc_raw_subdir:
        tc_raw_dir = microstate_path
        microstate_path = microstate_path.parent
    if not tc_raw_dir.is_dir():
        raise FileNotFoundError(f"raw TeraChem directory not found: {tc_raw_dir}")

    resp_dir = microstate_path / "terachem" / "respout"
    esp_dir = microstate_path / "terachem" / "espxyz"
    resp_dir.mkdir(parents=True, exist_ok=True)
    esp_dir.mkdir(parents=True, exist_ok=True)

    frames = _find_frame_dirs(tc_raw_dir)
    if not frames:
        raise FileNotFoundError(f"No frame directories found under {tc_raw_dir}")

    counter = 1
    copied_resp = 0
    copied_esp = 0
    missing_resp = 0
    missing_esp = 0

    for frame in frames:
        splits = frame / f"scr.carved_{frame.name}" / "splits"
        confs = [p for p in splits.iterdir() if p.is_dir()]
        confs = sorted(confs, key=lambda p: _natural_key(p.name))
        for conf in confs:
            resp = conf / "resp-vac.out"
            esp = conf / "scr.nowater" / "esp.xyz"
            tag = f"{counter:04d}"

            if resp.is_file():
                (resp_dir / f"conf{tag}.resp.out").write_bytes(resp.read_bytes())
                copied_resp += 1
            else:
                missing_resp += 1

            if esp.is_file():
                (esp_dir / f"conf{tag}.esp.xyz").write_bytes(esp.read_bytes())
                copied_esp += 1
            else:
                missing_esp += 1

            counter += 1

    return {
        "microstate": microstate_path.name,
        "tc_raw_dir": tc_raw_dir,
        "resp_dir": resp_dir,
        "esp_dir": esp_dir,
        "frames": len(frames),
        "confs": counter - 1,
        "copied_resp": copied_resp,
        "copied_esp": copied_esp,
        "missing_resp": missing_resp,
        "missing_esp": missing_esp,
    }


def _find_esp_from_conf(conf_dir: Path, rst7_path: Path) -> Path | None:
    stem = rst7_path.stem
    candidates = [
        conf_dir / f"scr.{stem}" / "esp.xyz",
        conf_dir / f"scr.{rst7_path.name}" / "esp.xyz",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    for cand in conf_dir.glob("scr.*/esp.xyz"):
        if cand.is_file():
            return cand
    return None


def process_tc_resp_runs(
    microstate_path: Path,
    confs_subdir: str = "input_tc_structures/confs",
) -> Dict[str, object]:
    microstate_path = microstate_path.resolve()
    confs_root = microstate_path / confs_subdir
    if not confs_root.is_dir():
        raise FileNotFoundError(f"Missing confs directory: {confs_root}")

    resp_dir = microstate_path / "terachem" / "respout"
    esp_dir = microstate_path / "terachem" / "espxyz"
    resp_dir.mkdir(parents=True, exist_ok=True)
    esp_dir.mkdir(parents=True, exist_ok=True)

    conf_dirs = sorted([p for p in confs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    counter = 1
    copied_resp = 0
    copied_esp = 0
    missing_resp = 0
    missing_esp = 0

    for conf_dir in conf_dirs:
        resp = conf_dir / "resp.out"
        rst7 = conf_dir / "nowater.rst7"
        if not rst7.exists():
            rst7_candidates = list(conf_dir.glob("*.rst7"))
            if rst7_candidates:
                rst7 = rst7_candidates[0]
        esp = _find_esp_from_conf(conf_dir, rst7)

        tag = f"{counter:04d}"

        if resp.is_file():
            (resp_dir / f"conf{tag}.resp.out").write_bytes(resp.read_bytes())
            copied_resp += 1
        else:
            missing_resp += 1

        if esp and esp.is_file():
            (esp_dir / f"conf{tag}.esp.xyz").write_bytes(esp.read_bytes())
            copied_esp += 1
        else:
            missing_esp += 1

        counter += 1

    return {
        "microstate": microstate_path.name,
        "confs_root": confs_root,
        "resp_dir": resp_dir,
        "esp_dir": esp_dir,
        "confs": counter - 1,
        "copied_resp": copied_resp,
        "copied_esp": copied_esp,
        "missing_resp": missing_resp,
        "missing_esp": missing_esp,
    }
