from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.symmetry.symmetry import buckets_from_pdb


def find_microstate_dir(root: Path, microstate: str) -> Path:
    """Return the directory for the requested microstate."""

    microstate_dir = root / "data" / "microstates" / microstate.upper()
    if not microstate_dir.is_dir():
        raise FileNotFoundError(f"Unknown microstate '{microstate}'. Expected directory at {microstate_dir}")
    return microstate_dir


def resolve_pdb_path(microstate_dir: Path, microstate: str) -> Path:
    """Locate the PDB file for the given microstate."""

    expected = microstate_dir / f"{microstate.lower()}.pdb"
    if expected.exists():
        return expected

    pdb_files = list(microstate_dir.glob("*.pdb"))
    if len(pdb_files) == 1:
        return pdb_files[0]

    if not pdb_files:
        raise FileNotFoundError(f"No PDB file found in {microstate_dir}")

    candidates = ", ".join(f.name for f in pdb_files)
    raise FileNotFoundError(
        f"Could not determine microstate PDB file. Provide one of: {candidates}"
    )


def _format_bucket(bucket: Iterable[int]) -> str:
    """Return a compact string representation like [0,1,2]."""

    values = ",".join(str(idx) for idx in bucket)
    return f"[{values}]"


def write_buckets(path: Path, buckets: Iterable[Iterable[int]]) -> None:
    """Persist the symmetry bucket list in a compact, readable format."""

    serialized = [_format_bucket(bucket) for bucket in buckets]
    content = "[\n" + ",\n".join(serialized) + "\n]\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WL symmetry buckets for a microstate PDB.")
    parser.add_argument(
        "--microstate",
        required=True,
        help="Microstate identifier (e.g., APP). Case-insensitive.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        "--r",
        type=int,
        default=10,
        help="Weisfeiler-Lehman refinement radius (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the bucket list. Defaults to data/microstates/<microstate>/symmetry-buckets/r<R>.dat",
    )
    parser.add_argument(
        "--remove-hs",
        action="store_true",
        help="Remove hydrogens before WL refinement (default keeps them).",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    microstate_dir = find_microstate_dir(repo_root, args.microstate)
    pdb_path = resolve_pdb_path(microstate_dir, args.microstate)

    buckets: List[List[int]] = buckets_from_pdb(
        pdb_path,
        radius=args.radius,
        remove_hs=args.remove_hs,
    )

    output_path = args.output
    if output_path is None:
        output_dir = microstate_dir / "symmetry-buckets"
        output_path = output_dir / f"r{args.radius}.dat"

    write_buckets(output_path, buckets)

    rel_path = output_path.relative_to(repo_root)
    print(f"Wrote {len(buckets)} buckets (radius={args.radius}) to {rel_path}")


if __name__ == "__main__":
    main()
