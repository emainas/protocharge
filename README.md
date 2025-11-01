# biliresp

<p>
  <img src="docs/img/profile.png" alt="Electrostatic potential for biliverdin" width="200">
</p>

<p>
  <a href="https://github.com/emainas/biliresp/actions">
    <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha">
  </a>
  <a href="https://github.com/emainas/biliresp/actions/workflows/tests.yml">
    <img src="https://github.com/emainas/biliresp/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
</p>

> **Status:** pre-release, under active development. Interfaces may change without notice.

Utilities for parsing electrostatic potential output (In [TeraChem](http://www.petachem.com/products.html) this is included in `resp.out`) and ESP grid (In [TeraChem](http://www.petachem.com/products.html) this is outputed as `esp.xyz`) files. The package supplies:

- 📄 A parser (`resp.ParseRespDotOut`) for extracting RESP frames and ESP grids from an ab initio Molecular Dynamics trajectory or QM/MM trajectory (or a single conformer calculation can be used).
- 🧮 A linear ESP charge fitting implementation (`linearESPcharges.linear`).
- 📊 Dipole post-processing helpers (see `scripts/print_dipoles.py` and `tests/test_dipole.py`) and a mass-weighted center-of-mass calculator that reuses xyz element ordering.
- 🛠️ Command-line entry points in `scripts/` for quick comparisons.

## Documentation

Full documentation, including guides and API references, lives on the project site: [biliresp Docs](https://emainas.github.io/biliresp/getting-started/).

## Quick start 🚀

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### Conda environment

If you prefer Conda, the repository ships with an environment specification that
matches the `biliresp` environment used during development:

```bash
conda env create -f environment.yml
conda activate biliresp
python -m pip install -e .
```

This environment pulls in `numpy`, `networkx`, and `rdkit` from `conda-forge`
and leaves the project itself managed through `pip install -e .` so that edits in
`src/` are immediately available.

## Run the test suite ✅

```bash
PYTHONPATH=src pytest -s tests
```

Use `-k` to narrow to an individual module while iterating (for example `-k test_dipole`).

## Command-line entry points ⚙️

All commands assume the sample RESP outputs in `data/raw/` and 78 atoms; adjust to your system as needed.

```bash
# Compare RESP ESP charges to fitted charges
python scripts/compare_charges.py data/raw/resp.out data/raw/esp.xyz 78 --frame -1

# Print QM, ESP, and fitted dipoles for a frame
python scripts/print_dipoles.py data/raw/resp.out data/raw/esp.xyz data/raw/1.pose.xyz 78 --frame -1
```

Both scripts accept `--help` for a summary of arguments.

## Multiconfigurational RESP on HPC (Slurm) 🖥️

The multiconfiguration workflow builds very large Coulomb matrices. For production fits we
recommend running on a Slurm cluster:

1. **Precompute the stacked matrices locally (optional).**
   From your workstation, generate the Coulomb matrix `A` and ESP vector `Y` once:

   ```bash
   python -m multiconfresp.mcresp --microstate <MICROSTATE> --save
   ```

   This creates `data/microstates/<microstate>/multiconfRESP/coulomb_matrix.npz` and
   `esp_vector.npz`. Copy the relevant `data/` subtree and `src/` directory to your cluster
   workspace (for example with `rsync` or `scp`).

2. **Create the Conda environment on the cluster.**

   ```bash
   conda env create -f environment.yml
   conda activate biliresp
   python -m pip install -e .
   ```

   The environment depends on `numpy`, `scipy`, `PyYAML`, `rdkit`, and friends. Installing
   from `environment.yml` ensures compatible versions on the HPC nodes.

3. **Submit the Slurm job.**

   A ready-to-edit submission script lives in `scripts/run_multiconfresp.slurm`.
   Update the email address, microstate name, and resource requests as needed, then
   submit:

   ```bash
   sbatch scripts/run_multiconfresp.slurm
   ```

   By default the script runs

   ```bash
   python -m multiconfresp.mcresp --microstate <MICROSTATE> --load-and-resp --maxiter 400
   ```

   and emails you if the job fails.

4. **Collect results.**
   The RESP step-one and final charges, along with JSON loggers for both RESP passes,
   are written to `data/microstates/<microstate>/multiconfRESP/` on the cluster.
   Copy them back to your workstation after the job finishes.

## Under development 🧪

1. Restraint ESP charges
2. Symmetry-adapted regression
3. Multiconformational RESP charges
