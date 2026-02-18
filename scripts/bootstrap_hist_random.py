#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from protocharge.training.linearESPcharges.linear import prepare_linear_system

MICROSTATES = ["HIP", "HIE", "HID"]
N_BLOCKS = 10  # blocks (frames) per microstate
BLOCK_SIZE = 191  # configs per block/frame


def index_configs(root: Path) -> List[str]:
    resp_dir = root / "terachem" / "respout"
    esp_dir = root / "terachem" / "espxyz"
    resp = {p.stem.replace(".resp", "") for p in resp_dir.glob("*.resp.out")}
    esp = {p.stem.replace(".esp", "") for p in esp_dir.glob("*.esp.xyz")}
    common = sorted(resp & esp)
    if not common:
        raise FileNotFoundError(f"No matching resp.out / esp.xyz in {root}")
    return common


def blocks_from_stems(stems: List[str], n_blocks: int) -> List[List[str]]:
    """Split stems into consecutive blocks."""
    blocks: List[List[str]] = []
    per_block = len(stems) // n_blocks
    for i in range(n_blocks):
        start = i * per_block
        end = (i + 1) * per_block if i < n_blocks - 1 else len(stems)
        blocks.append(stems[start:end])
    return blocks


def stack_configs(root: Path, stems: List[str], pdb_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    As: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    for stem in stems:
        resp_path = root / "terachem" / "respout" / f"{stem}.resp.out"
        esp_path = root / "terachem" / "espxyz" / f"{stem}.esp.xyz"
        A, V, _, _ = prepare_linear_system(resp_path, esp_path, pdb_atoms, frame_index=-1, grid_frame_index=0)
        As.append(A)
        Ys.append(V)
    design_matrix = np.vstack(As)
    esp_values = np.concatenate([v.reshape(-1) for v in Ys])
    return design_matrix, esp_values


def write_matrices(out_root: Path, design: np.ndarray, esp: np.ndarray) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_root / "coulomb_matrix.npz", A=design)
    np.savez_compressed(out_root / "esp_vector.npz", Y=esp)


def write_manifest(template: Path, out_path: Path, rep_root: Path, base_dir: Path) -> None:
    import yaml as _yaml

    manifest = _yaml.safe_load(template.read_text(encoding="utf-8"))
    global_buckets = manifest.get("global_buckets")
    for mol in manifest.get("molecules", []):
        name = mol["name"]
        base_root = base_dir / name
        mol["root"] = str(base_root.resolve())
        mol["coulomb"] = str((rep_root / name / "multiconfRESP" / "coulomb_matrix.npz").resolve())
        mol["esp"] = str((rep_root / name / "multiconfRESP" / "esp_vector.npz").resolve())
    if global_buckets:
        gb_path = Path(global_buckets)
        if not gb_path.is_absolute():
            gb_path = (template.parent / gb_path).resolve()
        manifest["global_buckets"] = str(gb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_yaml.safe_dump(manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Random bootstrap HIP/HIE/HID configs into stacked matrices (with replacement).")
    parser.add_argument("--replicate", type=int, required=True, help="Replicate id (used to seed RNG).")
    parser.add_argument("--base-dir", type=Path, default=Path("input/microstates"), help="Base microstate dir.")
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/bootstrap_hist_random"), help="Output base.")
    parser.add_argument("--manifest-template", type=Path, default=Path("configs/manifest_hist.yaml"))
    args = parser.parse_args()

    rng = random.Random(args.replicate)

    blocks: Dict[str, List[List[str]]] = {}
    pdb_counts: Dict[str, int] = {}
    for ms in MICROSTATES:
        root = args.base_dir / ms
        stems = index_configs(root)
        blocks[ms] = blocks_from_stems(stems, N_BLOCKS)
        pdb_path = root / f"{ms.lower()}.pdb"
        pdb_counts[ms] = sum(1 for line in pdb_path.read_text().splitlines() if line.startswith(("ATOM", "HETATM")))

    rep_root = args.output_dir / f"rep_{args.replicate:03d}"

    for ms in MICROSTATES:
        chosen_blocks = [blocks[ms][rng.randrange(N_BLOCKS)] for _ in range(N_BLOCKS)]
        stems = [s for blk in chosen_blocks for s in blk]
        design, esp = stack_configs(args.base_dir / ms, stems, pdb_counts[ms])
        write_matrices(rep_root / ms / "multiconfRESP", design, esp)

    write_manifest(args.manifest_template, rep_root / "manifest.yaml", rep_root, args.base_dir)


if __name__ == "__main__":
    main()
