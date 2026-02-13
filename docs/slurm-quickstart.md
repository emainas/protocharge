# Slurm Quickstart

This project provides a CLI that can submit RESP workflows to Slurm. The CLI reads a YAML config, builds the right command, writes a Slurm script under `results/slurm/`, and submits it via `sbatch`.

## 1) Prepare inputs

Place your inputs under `data/microstates/<MICROSTATE>/`:

- `<MICROSTATE>.pdb`
- `symmetry-buckets/r8.dat`
- `terachem/espxyz/*.esp.xyz`
- `terachem/respout/*.resp.out`

See the minimal example layout in `docs/examples/microstates/`.

If you only have raw TeraChem outputs under a `raw_terachem_outputs/` directory, use:

```bash
biliresp --process data/microstates/<MICROSTATE>
```

## 2) Create a config

Create `configs/<microstate>/<function>/config.yaml`. Example:

```yaml
microstate: HID
args:
  frame: -1
  maxiter: 400
slurm:
  time: "24:00:00"
  mem: "64G"
  cpus_per_task: 12
```

A complete example config is in `docs/examples/configs/`.

## 3) Dry run (recommended)

Use `--dry-run` to see the resolved command and config path before submitting:

```bash
biliresp --function twostepRESP_basic --yaml HID/twostepRESP_basic --dry-run
```

## 4) Submit to Slurm

```bash
biliresp --function twostepRESP_basic --yaml HID/twostepRESP_basic --slurm
```

This writes a script under `results/slurm/` and submits it with `sbatch`.

## Slurm settings

You can override these keys under `slurm:` in the config:

- `job_name`
- `output` / `error`
- `time`
- `nodes`
- `cpus_per_task`
- `mem`
- `partition`
- `account`
- `mail_type` / `mail_user`

The template loads `anaconda`, activates the `biliresp` conda env, and sets `PYTHONPATH` to the project `src/` directory.
