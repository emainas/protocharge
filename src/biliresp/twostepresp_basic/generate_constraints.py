from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class ResidueState:
    name: str | None
    protcnt: int | None
    charges: List[float]


@dataclass(frozen=True)
class ResidueDefinition:
    atom_labels: List[str]
    states: List[ResidueState]


def _literal_eval(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception as exc:  # pragma: no cover - best effort helper
        raise ValueError(f"Unable to literal_eval AST node {ast.dump(node)}") from exc


def load_residue_definition(path: Path, residue_name: str) -> ResidueDefinition:
    module = ast.parse(path.read_text(), filename=str(path))

    atom_labels: List[str] | None = None
    states: List[ResidueState] = []

    for node in ast.walk(module):
        # Capture ``RES = TitratableResidue('PAT', [...])``
        if isinstance(node, ast.Assign):
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not isinstance(func, ast.Name):
                continue
            if func.id != "TitratableResidue":
                continue
            if not call.args:
                continue
            residue_id = _literal_eval(call.args[0])
            if residue_id != residue_name:
                continue
            if len(call.args) < 2:
                raise ValueError(
                    f"Residue {residue_name} definition in {path} does not provide atom labels."
                )
            atom_arg = call.args[1]
            atom_labels = [_literal_eval(elem) for elem in atom_arg.elts]  # type: ignore[attr-defined]

        # Capture ``PAT.add_state(... charges=[...])``
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            func = call.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name):
                continue
            if func.value.id != residue_name or func.attr != "add_state":
                continue

            keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            charges_node = keywords.get("charges")
            if charges_node is None:
                raise ValueError(
                    f"Encountered {residue_name}.add_state call without charges in {path}."
                )
            charges = [_literal_eval(elem) for elem in charges_node.elts]  # type: ignore[attr-defined]

            protcnt_node = keywords.get("protcnt")
            protcnt = _literal_eval(protcnt_node) if protcnt_node is not None else None

            # Try to capture inline comment (state name) from preceding node if available
            state_name = None
            if call.keywords:
                last_kw = call.keywords[-1]
                if isinstance(last_kw.value, ast.Constant) and isinstance(last_kw.value.value, str):
                    state_name = last_kw.value.value

            states.append(ResidueState(state_name, protcnt, charges))

    if atom_labels is None:
        raise ValueError(f"Residue {residue_name} not found in {path}")

    if not states:
        raise ValueError(f"No states discovered for residue {residue_name} in {path}")

    return ResidueDefinition(atom_labels=atom_labels, states=states)


def load_buckets(path: Path) -> List[List[int]]:
    return list(ast.literal_eval(path.read_text()))


def load_atom_labels_from_pdb(path: Path) -> List[str]:
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                labels.append(line[12:16].strip())
    return labels


def build_bucket_constraints(
    buckets: Sequence[Sequence[int]],
    atom_labels: Sequence[str],
    charge_by_label: Dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> Dict[int, Dict[str, Iterable[str] | float]]:
    constraints: Dict[int, Dict[str, Iterable[str] | float]] = {}

    for bucket_index, bucket in enumerate(buckets):
        names = [atom_labels[idx] for idx in bucket]

        try:
            charges = [charge_by_label[name] for name in names]
        except KeyError:
            continue  # Skip buckets that are not part of the residue of interest

        if not charges:
            continue

        max_charge = max(charges)
        min_charge = min(charges)
        if abs(max_charge - min_charge) > tolerance:
            raise ValueError(
                f"Inconsistent target charges for bucket {bucket_index} with atoms {names}: {charges}"
            )

        constraints[bucket_index] = {"labels": names, "value": float(charges[0])}

    return constraints


def render_yaml(total_charge: float, constraints: Dict[int, Dict[str, Iterable[str] | float]]) -> str:
    lines = ["total_charge: {:.6f}".format(total_charge), "bucket_constraints:"]
    for bucket_index in sorted(constraints):
        entry = constraints[bucket_index]
        labels = entry["labels"]
        value = entry["value"]
        label_list = ", ".join(str(label) for label in labels)
        lines.append(f"  - bucket: {bucket_index}")
        lines.append(f"    labels: [{label_list}]")
        lines.append(f"    value: {value:.6f}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate symmetry-bucket charge constraints.")
    parser.add_argument("--pdb", type=Path, required=True, help="PDB file with atom labels.")
    parser.add_argument("--buckets", type=Path, required=True, help="Symmetry bucket file.")
    parser.add_argument("--parmed", type=Path, required=True, help="ParmEd residue definition file.")
    parser.add_argument("--residue", type=str, default="PAT", help="Residue identifier to extract.")
    parser.add_argument(
        "--state-index",
        type=int,
        default=0,
        help="Index of the protonation state (among those matching protcnt==1) to use.",
    )
    parser.add_argument(
        "--protcnt",
        type=int,
        default=1,
        help="Filter states by proton count before applying state-index.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination YAML file.")
    parser.add_argument(
        "--total-charge",
        type=float,
        required=True,
        help="Total charge constraint for the system (applied to bucket space).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Allowed charge variability within a bucket.",
    )

    args = parser.parse_args(argv)

    residue = load_residue_definition(args.parmed, args.residue)

    candidate_states = [
        state for state in residue.states if args.protcnt is None or state.protcnt == args.protcnt
    ]
    if not candidate_states:
        raise ValueError(
            f"No states found for residue {args.residue} with protcnt={args.protcnt} in {args.parmed}"
        )
    if args.state_index < 0 or args.state_index >= len(candidate_states):
        raise IndexError(
            f"state_index {args.state_index} out of range for {len(candidate_states)} matching states"
        )
    target_state = candidate_states[args.state_index]

    if len(target_state.charges) != len(residue.atom_labels):
        raise ValueError(
            f"Charges length ({len(target_state.charges)}) does not match atom label count ({len(residue.atom_labels)})"
        )

    charge_by_label = {
        label: float(charge) for label, charge in zip(residue.atom_labels, target_state.charges)
    }

    buckets = load_buckets(args.buckets)
    atom_labels = load_atom_labels_from_pdb(args.pdb)

    max_index = max(idx for bucket in buckets for idx in bucket)
    if max_index >= len(atom_labels):
        raise IndexError(
            f"Bucket references atom index {max_index} but PDB only contains {len(atom_labels)} atoms."
        )

    bucket_constraints = build_bucket_constraints(
        buckets, atom_labels, charge_by_label, tolerance=args.tolerance
    )

    yaml_text = render_yaml(args.total_charge, bucket_constraints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml_text, encoding="utf-8")


if __name__ == "__main__":
    main()
