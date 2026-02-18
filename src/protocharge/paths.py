from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root relative to this source file."""
    return Path(__file__).resolve().parents[2]


def input_root() -> Path:
    return project_root() / "input"


def output_root() -> Path:
    return project_root() / "output"


def configs_root() -> Path:
    return project_root() / "configs"


def microstate_input_root(microstate: str) -> Path:
    return input_root() / "microstates" / microstate


def microstate_config_root(microstate: str) -> Path:
    return configs_root() / microstate


def microstate_constraints_root(microstate: str) -> Path:
    return microstate_config_root(microstate) / "charge-contraints"


def microstate_output_root(microstate: str) -> Path:
    return output_root() / microstate


def ensure_results_dir(microstate: str, *parts: str) -> Path:
    path = microstate_output_root(microstate).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Backward-compatible aliases (deprecated)
def data_root() -> Path:
    return input_root()


def results_root() -> Path:
    return output_root()


def microstate_results_root(microstate: str) -> Path:
    return microstate_output_root(microstate)
