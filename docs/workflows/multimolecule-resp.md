# Multi-Molecule RESP

The multi-molecule workflow (`protocharge.training.multimoleculeresp.mmresp`) fits charges across multiple microstates at once. It supports global symmetry buckets that tie atoms across molecules, optional frozen buckets, and per-molecule total or group constraints.

## Run from a Manifest

```bash
python -m protocharge.training.multimoleculeresp.mmresp \
  --manifest configs/manifest_hist.yaml \
  --output output/mmresp_hist/charges.npz \
  --ridge 1e-6
```

The manifest defines:

- A list of molecules, each with paths to Coulomb matrices and ESP vectors.
- Global symmetry buckets (optional), including target charges or frozen values.
- Per-molecule constraints (total or group constraints).

## Bootstrap Workflows

Two helper scripts generate bootstrap matrices and manifests:

- `scripts/bootstrap_hist.py` – deterministic leave-two-out subsets.
- `scripts/bootstrap_hist_random.py` – random resampling with replacement.

These are paired with the `run_bootstrap_*.slurm` scripts for cluster runs.
