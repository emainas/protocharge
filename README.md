# biliresp

[![Release](https://img.shields.io/github/v/release/emainas/biliresp)](https://github.com/emainas/biliresp/releases/latest)
[![Tests](https://github.com/emainas/biliresp/actions/workflows/tests.yml/badge.svg)](https://github.com/emainas/biliresp/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://emainas.github.io/biliresp/getting-started/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Utilities for parsing TeraChem RESP/ESP outputs, fitting electrostatic potential (ESP) charges, and validating dipoles. The project provides RESP/ESP parsing, linear and hyperbolic solvers, two-step workflows, and ensemble/multi-molecule pipelines.

## Documentation

Project documentation (workflows + API notes): https://emainas.github.io/biliresp/getting-started/

## Install

```bash
conda env create -f environment.yml
conda activate biliresp
python -m pip install -e .
```

## Tests

```bash
pytest -q
```

## CLI (recommended)

Create a YAML config under `configs/<microstate>/<function>/config.yaml` and run:

```bash
biliresp --function twostepRESP_basic --yaml HID/twostepRESP_basic --slurm
```

`--yaml` can be a full path or a `configs/` subpath. With `--slurm`, the CLI generates a Slurm script and submits it.

## Project layout

- `data/` holds inputs and parameters tied to a microstate (PDB, esp.xyz, resp.out, symmetry buckets).
- `configs/` holds YAML run configurations and charge-constraint files.
- `results/` holds outputs organized by microstate and function.

See `docs/` and `docs/workflows/` for the full set of workflows and variants.
