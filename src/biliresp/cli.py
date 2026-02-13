from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

from biliresp.paths import data_root, project_root, results_root
from biliresp.run_tc_resp import run_tc_resp
from biliresp.terachem_processing import process_tc_resp_runs, process_terachem_outputs


def _default_microstate_root(microstate: str) -> Path:
    return data_root() / "microstates" / microstate


def _slurm_header(slurm: Dict[str, str]) -> List[str]:
    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm.get('job_name', 'biliresp')}",
        f"#SBATCH --output={slurm.get('output', 'slurm_logs/%x.%j.out')}",
        f"#SBATCH --error={slurm.get('error', 'slurm_logs/%x.%j.err')}",
        f"#SBATCH --time={slurm.get('time', '24:00:00')}",
        f"#SBATCH --nodes={slurm.get('nodes', 1)}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 12)}",
        f"#SBATCH --mem={slurm.get('mem', '64G')}",
    ]
    if "partition" in slurm:
        header.append(f"#SBATCH --partition={slurm['partition']}")
    if "account" in slurm:
        header.append(f"#SBATCH --account={slurm['account']}")
    if "mail_type" in slurm:
        header.append(f"#SBATCH --mail-type={slurm['mail_type']}")
    if "mail_user" in slurm:
        header.append(f"#SBATCH --mail-user={slurm['mail_user']}")
    return header


def _render_slurm_script(command: List[str], slurm: Dict[str, str]) -> str:
    lines = _slurm_header(slurm)
    lines += [
        "",
        "set -euo pipefail",
        "set -x",
        "",
        f"PROJECT_SRC=\"{project_root() / 'src'}\"",
        "export PYTHONPATH=\"$PROJECT_SRC${PYTHONPATH:+:$PYTHONPATH}\"",
        "",
        "module load anaconda || true",
        "eval \"$(conda shell.bash hook)\"",
        "conda activate biliresp",
        "",
        "cd \"$SLURM_SUBMIT_DIR\"",
        "mkdir -p slurm_logs",
        "",
        "srun " + " ".join(command),
        "",
    ]
    return "\n".join(lines)


def _args_to_cli(args: Dict[str, object]) -> List[str]:
    cli: List[str] = []
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                cli.extend([flag, str(item)])
            continue
        if value is None:
            continue
        cli.extend([flag, str(value)])
    return cli


FUNCTION_ALIASES = {
    "rawESP": "raw-esp",
    "rawESP_constraints": "raw-esp-constraints",
    "rawESP_multi": "raw-esp-multi",
    "onestepRESP": "onestep-resp",
    "twostepRESP_basic": "twostep-basic",
    "twostepRESP_masked_total": "twostep-masked-total",
    "twostepRESP_group_constraints": "twostep-group",
    "twostepRESP_frozen_buckets": "twostep-frozen",
    "multiconfRESP": "multiconf",
    "multiconfRESP_reduced_basic": "reduced-basic",
    "multiconfRESP_reduced_masked_total": "reduced-masked-total",
    "multiconfRESP_reduced_group_constraints": "reduced-group",
    "multimolecule": "multimolecule",
}


def _normalize_function(name: str) -> str:
    return FUNCTION_ALIASES.get(name, name)


def _command_for_function(function: str, cfg: Dict[str, object]) -> Tuple[List[str], Path | None]:
    microstate = cfg.get("microstate")
    args: Dict[str, object] = dict(cfg.get("args", {}) or {})

    # Map function names to the legacy scripts/modules.
    function = _normalize_function(function)

    if function == "raw-esp":
        script = project_root() / "scripts" / "generate_raw_esp_matrix.py"
        if microstate:
            args.setdefault("microstate_root", _default_microstate_root(microstate))
        return [sys.executable, str(script), *_args_to_cli(args)], None
    if function == "raw-esp-constraints":
        script = project_root() / "scripts" / "generate_raw_esp_matrix_2.py"
        if microstate:
            args.setdefault("microstate_root", _default_microstate_root(microstate))
        return [sys.executable, str(script), *_args_to_cli(args)], None
    if function == "raw-esp-multi":
        script = project_root() / "scripts" / "generate_raw_esp_matrix_3.py"
        if microstate:
            args.setdefault("microstate_root", _default_microstate_root(microstate))
        return [sys.executable, str(script), *_args_to_cli(args)], None
    if function == "onestep-resp":
        script = project_root() / "scripts" / "generate_resp_charges_matrix.py"
        if microstate:
            args.setdefault("microstate_root", _default_microstate_root(microstate))
        return [sys.executable, str(script), *_args_to_cli(args)], None
    if function == "twostep-basic":
        script = project_root() / "scripts" / "generate_twostep_resp.py"
        if microstate:
            args.setdefault("microstate_root", _default_microstate_root(microstate))
        return [sys.executable, str(script), *_args_to_cli(args)], None
    if function == "twostep-masked-total":
        script = project_root() / "scripts" / "generate_twostep_resp_masked_total.py"
        if microstate:
            args.setdefault("microstate_root", _default_microstate_root(microstate))
        return [sys.executable, str(script), *_args_to_cli(args)], None
    if function == "twostep-group":
        script = project_root() / "scripts" / "generate_twostep_resp_group_constraints.py"
        if microstate:
            args.setdefault("microstate_root", _default_microstate_root(microstate))
        return [sys.executable, str(script), *_args_to_cli(args)], None
    if function == "twostep-frozen":
        script = project_root() / "scripts" / "generate_twostep_resp_frozen_buckets.py"
        if microstate:
            args.setdefault("microstate_root", _default_microstate_root(microstate))
        return [sys.executable, str(script), *_args_to_cli(args)], None
    if function == "multiconf":
        module = ["-m", "biliresp.multiconfresp.mcresp"]
        if microstate:
            args.setdefault("microstate", microstate)
        return [sys.executable, *module, *_args_to_cli(args)], None
    if function == "reduced-basic":
        module = ["-m", "biliresp.reduced_basic.reduced"]
        if microstate:
            args.setdefault("microstate", microstate)
        return [sys.executable, *module, *_args_to_cli(args)], None
    if function == "reduced-masked-total":
        module = ["-m", "biliresp.reduced_masked_total.reduced"]
        if microstate:
            args.setdefault("microstate", microstate)
        return [sys.executable, *module, *_args_to_cli(args)], None
    if function == "reduced-group":
        module = ["-m", "biliresp.reduced_group_constraints.reduced"]
        if microstate:
            args.setdefault("microstate", microstate)
        return [sys.executable, *module, *_args_to_cli(args)], None
    if function == "multimolecule":
        module = ["-m", "biliresp.multimoleculeresp.mmresp"]
        return [sys.executable, *module, *_args_to_cli(args)], None

    raise ValueError(f"Unknown function '{function}'")


PATH_ARG_KEYS = {
    "microstate_root",
    "bucket_file",
    "total_constraint",
    "bucket_constraints",
    "mask_step1",
    "mask_step2",
    "group_constraint",
    "frozen_buckets",
    "resp_out",
    "esp_xyz",
    "pdb",
    "output",
    "manifest",
    "base_dir",
    "output_dir",
    "manifest_template",
}


def _resolve_paths_in_args(args: Dict[str, object], base_dir: Path) -> Dict[str, object]:
    resolved: Dict[str, object] = {}
    for key, value in args.items():
        if key in PATH_ARG_KEYS:
            if isinstance(value, str):
                candidate = Path(value)
                resolved[key] = (
                    candidate if candidate.is_absolute() else (base_dir / candidate)
                )
                continue
            if isinstance(value, list):
                items: List[Path] = []
                for item in value:
                    if not isinstance(item, str):
                        items.append(item)
                        continue
                    candidate = Path(item)
                    items.append(candidate if candidate.is_absolute() else (base_dir / candidate))
                resolved[key] = items
                continue
        resolved[key] = value
    return resolved


def _load_config(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping.")
    args = data.get("args")
    if isinstance(args, dict):
        data["args"] = _resolve_paths_in_args(args, path.parent)
    return data


def _resolve_config_path(path_or_name: str, function: str | None = None) -> Path:
    candidate = Path(path_or_name)
    if candidate.is_file():
        return candidate.resolve()
    # Treat as configs/<name>.yaml or configs/<name>/config.yaml
    configs_root = project_root() / "configs"
    direct = configs_root / f"{path_or_name}.yaml"
    if direct.is_file():
        return direct.resolve()
    nested = configs_root / path_or_name / "config.yaml"
    if nested.is_file():
        return nested.resolve()
    if function:
        func_candidates = [
            function,
            _normalize_function(function),
        ]
        for func_name in dict.fromkeys(func_candidates):
            by_func = configs_root / path_or_name / func_name / "config.yaml"
            if by_func.is_file():
                return by_func.resolve()
            by_func_alt = configs_root / path_or_name / f"{func_name}.yaml"
            if by_func_alt.is_file():
                return by_func_alt.resolve()
    raise FileNotFoundError(f"Config not found: {path_or_name}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run biliresp workflows locally or via Slurm.")
    parser.add_argument("--function", help="Workflow function name.")
    parser.add_argument("--yaml", dest="yaml_path", help="YAML config path or name under configs/.")
    parser.add_argument("--process", dest="process_path", help="Process raw TeraChem outputs under a microstate path.")
    parser.add_argument(
        "--tc-raw-subdir",
        default="raw_terachem_outputs",
        help="Subdirectory name that holds raw TeraChem outputs.",
    )
    parser.add_argument("--run-tc-resp", dest="tc_resp_path", help="Prepare and submit TeraChem RESP jobs.")
    parser.add_argument(
        "--process-tc-resp",
        dest="process_tc_resp_path",
        help="Collect TeraChem RESP outputs from input_tc_structures and write to terachem/ folders.",
    )
    parser.add_argument("--slurm", action="store_true", help="Submit as a Slurm job.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command and exit.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.process_path:
        summary = process_terachem_outputs(Path(args.process_path), tc_raw_subdir=args.tc_raw_subdir)
        print(
            "Processed",
            summary["confs"],
            "configs from",
            summary["frames"],
            "frames.",
        )
        print("RESP:", summary["resp_dir"])
        print("ESP:", summary["esp_dir"])
        if summary["missing_resp"] or summary["missing_esp"]:
            print(
                "Missing files:",
                f"resp={summary['missing_resp']}",
                f"esp={summary['missing_esp']}",
            )
        return

    if args.tc_resp_path:
        results = run_tc_resp(Path(args.tc_resp_path), submit=True)
        print(f"Submitted {len(results)} TeraChem RESP jobs.")
        return

    if args.process_tc_resp_path:
        summary = process_tc_resp_runs(Path(args.process_tc_resp_path))
        print(f"Collected {summary['copied_resp']} resp.out and {summary['copied_esp']} esp.xyz files.")
        if summary["missing_resp"] or summary["missing_esp"]:
            print(
                "Missing files:",
                f"resp={summary['missing_resp']}",
                f"esp={summary['missing_esp']}",
            )
        return

    if not args.function or not args.yaml_path:
        raise SystemExit("Provide --function and --yaml, or use --process.")

    config_path = _resolve_config_path(args.yaml_path, args.function)
    cfg = _load_config(config_path)

    function = args.function
    command, _ = _command_for_function(function, cfg)

    if args.dry_run:
        print("Command:", " ".join(command))
        print("Config:", str(config_path))
        return

    if args.slurm:
        slurm_cfg = dict(cfg.get("slurm", {}) or {})
        slurm_cfg.setdefault("job_name", f"{function}_{cfg.get('microstate', 'job')}")
        script_body = _render_slurm_script(command, slurm_cfg)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_dir = results_root() / "slurm"
        slurm_dir.mkdir(parents=True, exist_ok=True)
        script_path = slurm_dir / f"{function}_{cfg.get('microstate', 'job')}_{stamp}.slurm"
        script_path.write_text(script_body, encoding="utf-8")
        subprocess.run(["sbatch", str(script_path)], check=True)
        print(f"Submitted Slurm job via {script_path}")
        return

    subprocess.run(command, check=True)
