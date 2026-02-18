# protocharge

<!-- [![Release](https://img.shields.io/github/v/release/emainas/protocharge)](https://github.com/emainas/protocharge/releases/latest) -->
[![Tests](https://github.com/emainas/protocharge/actions/workflows/tests.yml/badge.svg)](https://github.com/emainas/protocharge/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://emainas.github.io/protocharge/getting-started/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Tools for generating ESP datasets, fitting RESP charges, and validating charge models.

## Documentation

Project documentation (workflows + API notes): https://emainas.github.io/protocharge/getting-started/

## Install

```bash
conda env create -f environment.yml
conda activate protocharge
python -m pip install -e .
```

## Tests

```bash
pytest -q
```

## Demo (MEA)

Step 1 — **MD prep** (build topology/coordinates with tleap):

```bash
pc --generator md.prep --yaml configs/MEA/generator.yaml
```

Uses:
- `input/microstates/MEA/initial-files/mea.mol2`
- `input/microstates/MEA/initial-files/mea.frcmod`

Writes:
- `output/MEA/md/prep/tleap.in`
- `output/MEA/md/prep/MEA.parm7`
- `output/MEA/md/prep/MEA.rst7`

Step 2 — **MD run** (generate MD inputs + submit with Slurm):

```bash
pc --generator md.run --yaml configs/MEA/generator.yaml --slurm
```

Uses:
- `output/MEA/md/prep/MEA.parm7`
- `output/MEA/md/prep/MEA.rst7`

Writes:
- `output/MEA/md/run/min.in`
- `output/MEA/md/run/heat.in`
- `output/MEA/md/run/equil-nvt.in`
- `output/MEA/md/run/equil-npt.in`
- `output/MEA/md/run/run.sh`
- `output/MEA/md/run/slurm.sh`

## Project layout

- `input/` holds inputs and parameters tied to a microstate (PDB, esp.xyz, resp.out, symmetry buckets).
- `configs/` holds YAML run configurations and charge-constraint files.
- `output/` holds outputs organized by microstate and function.

See `docs/` and `docs/workflows/` for the full set of workflows and variants.
