# Getting Started

## Prerequisites

- Python 3.9+ (conda recommended, includes RDKit).

## Install

```bash
conda env create -f environment.yml
conda activate protocharge
python -m pip install -e .
```

## Run the test suite

Run pytest from the repository root:

```bash
pytest -q
```

To run a single test module:

```bash
pytest -q tests/test_dipole.py
```

## Run a workflow

Create a config under `configs/<microstate>/<function>/config.yaml` and run:

```bash
pc --function twostepRESP_basic --yaml HID/twostepRESP_basic --slurm
```

`--yaml` accepts a full path or a `configs/` subpath. With `--slurm`, the CLI writes a Slurm script under `results/slurm/` and submits it via `sbatch`.

Use `--dry-run` to verify the resolved command without executing it:

```bash
pc --function twostepRESP_basic --yaml HID/twostepRESP_basic --dry-run
```

See `slurm-quickstart.md` for more detail.
