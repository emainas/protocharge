from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root relative to this source file."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return project_root() / "data"


def results_root() -> Path:
    return project_root() / "results"


def configs_root() -> Path:
    return project_root() / "configs"


def microstate_input_root(microstate: str) -> Path:
    return data_root() / "microstates" / microstate


def microstate_config_root(microstate: str) -> Path:
    return configs_root() / microstate


def microstate_constraints_root(microstate: str) -> Path:
    return microstate_config_root(microstate) / "charge-contraints"


def microstate_results_root(microstate: str) -> Path:
    return results_root() / microstate


def ensure_results_dir(microstate: str, *parts: str) -> Path:
    path = microstate_results_root(microstate).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
