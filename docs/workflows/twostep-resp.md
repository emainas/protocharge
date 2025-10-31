# Two-Step RESP Workflow

The two-step RESP pipeline reproduces the AMBER-style restrained electrostatic potential fit used by the TeraChem `DYES` protocol.[^amber-dyes] It applies the Weisfeiler–Lehman symmetry buckets computed for each microstate, enforces total and bucket-specific charge constraints, and reruns a second RESP optimisation that selectively relaxes atoms flagged in project-specific masks.

## Stages at a Glance

1. **Linear warm start** – `solve_least_squares_with_constraints` (in `twostepresp.tsresp`) fits bucket charges that satisfy the aggregate constraints and provides an initial KKT solution.
2. **Step-one RESP** – `resp_step` runs a Newton–Krylov solve with the hyperbolic restraint active on the mask defined in `mask_step_1.yaml`, producing `step1` charges for every configuration.
3. **Step-two RESP** – The solver reuses converged bucket charges but frees the subset flagged in `mask_step_2.yaml`, recomputing the fit while holding the remaining buckets fixed. The result is stored as `step2`, our recommended production charge set.

Both RESP steps log the KKT residual, gradient norm, and the charge vector, matching the diagnostics in `resp.resp`.

## Command-Line Entry Point

`scripts/generate_twostep_resp.py` batches the workflow across all TeraChem configurations in a microstate directory:

```bash
python scripts/generate_twostep_resp.py \
  --microstate-root data/microstates/PPP \
  --maxiter 400
```

The script automatically:

- Loads `symmetry-buckets/r8.dat` to build the expansion matrix.
- Reads total and bucket charge constraints from `charge-contraints/`.
- Applies step-one and step-two masks from YAML templates co-located in the microstate root.
- Persists `labels` (configuration stems), `step1`, and `step2` matrices to `twostepRESP/charges.npz`.

Use `--dry-run` to list the configuration stems without running RESP, or `--max-configs` to limit the number of fits during debugging.

## Retrying a Single Configuration

`scripts/retry_twostep_resp.py` mirrors the batch workflow for a single stem:

```bash
python scripts/retry_twostep_resp.py \
  --microstate-root data/microstates/PPP \
  --config conf2371 \
  --step1-maxiter 120 \
  --step2-maxiter 240 \
  --show-charges
```

The retrier reuses the masks, symmetry buckets, and constraints from the batch script, but exposes separate iteration limits for each stage and can print the atom-wise charges, making it ideal for diagnosing non-convergent fits.

## Consuming the Results

The resulting NPZ files provide configuration-major charge matrices. Pair them with the atom labels from the raw ESP workflow to slice by atom type, compare against single-step RESP, or feed into downstream notebooks such as `notebooks/twostepRESP.ipynb`. Because `step2` aligns with the AMBER two-stage procedure, it should be used as the primary source for production charge analyses unless diagnostics show step-two convergence issues.

[^amber-dyes]: AMBER DYES two-stage RESP protocol (`bibliography/amber-dyes.pdf`), bundled with the repository for reference.
