# Module Overview

This project is organized as a single package, `biliresp`, with focused submodules for parsing, fitting, and analysis. The table below summarizes the main modules and how they fit together.

| Module | Purpose | Key Entrypoints |
| --- | --- | --- |
| `biliresp.resp_parser` | Parse TeraChem `resp.out`, `esp.xyz`, and `.xyz` geometry frames. | `ParseRespDotOut`, `ParseESPXYZ`, `ParseDotXYZ` |
| `biliresp.linearESPcharges` | Build Coulomb matrices and solve the linear (unrestrained) ESP fit. | `prepare_linear_system`, `explicit_solution` |
| `biliresp.resp` | Single-frame RESP solver with hyperbolic restraint. | `fit_resp_charges`, `kkt_residual_at` |
| `biliresp.dipole` | Dipole and center-of-mass helpers. | `center_of_mass_bohr_from_xyz` |
| `biliresp.symmetry` | WL refinement and symmetry buckets for charge tying. | `buckets_from_pdb` |
| `biliresp.twostepresp_basic` | Two-step RESP with total + bucket constraints. | `resp_step`, `solve_least_squares_with_constraints` |
| `biliresp.twostepresp_masked_total` | Two-step RESP with a masked total-charge constraint. | `load_total_constraint`, `build_total_constraint_mask` |
| `biliresp.twostepresp_group_constraints` | Two-step RESP with group constraints (no implicit total). | `load_group_constraints` |
| `biliresp.twostepresp_frozen_buckets` | Two-step RESP with frozen bucket values. | `load_frozen_buckets` |
| `biliresp.multiconfresp` | Multi-configuration assembly for a microstate. | `mcresp.py` CLI |
| `biliresp.reduced_basic` | Reduced-space RESP solver (total + bucket constraints). | `reduced.py` CLI |
| `biliresp.reduced_masked_total` | Reduced-space solver with masked total constraint. | `reduced.py` CLI |
| `biliresp.reduced_group_constraints` | Reduced-space solver with group constraints. | `reduced.py` CLI |
| `biliresp.multimoleculeresp` | Multi-molecule, multi-config RESP (global buckets + optional freezing). | `mmresp.py` CLI |

For recipes and command-line workflows, see the pages under **Workflows**.
