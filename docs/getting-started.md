# Getting Started

## Prerequisites

- Python 3.9+ (conda recommended, includes RDKit).

## Install

```bash
conda env create -f environment.yml
conda activate biliresp
python -m pip install -e .
```

## Run the test suite

Run pytest from the repository root:

```bash
pytest -q
```

Use `-k` to narrow to a single test module when iterating, for example:

```bash
pytest -q tests/test_dipole.py
```

## Run a workflow

Create a config under `configs/<microstate>/<function>/config.yaml` and run:

```bash
biliresp --function twostepRESP_basic --yaml HID/twostepRESP_basic --slurm
```

`--yaml` accepts a full path or a `configs/` subpath. With `--slurm`, the CLI writes a Slurm script under `results/slurm/` and submits it via `sbatch`.

Use `--dry-run` to verify the resolved command without executing it:

```bash
biliresp --function twostepRESP_basic --yaml HID/twostepRESP_basic --dry-run
```

See `slurm-quickstart.md` for more detail.
