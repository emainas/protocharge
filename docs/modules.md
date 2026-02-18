# Module Overview

This project is organized as a single package, `protocharge`, with focused submodules for parsing, fitting, and analysis. The table below summarizes the main modules and how they fit together.

| Module | Purpose | Key Entrypoints |
| --- | --- | --- |
| `protocharge.generator.md_qmmm` | Generator pipeline for MD + QMMM dataset prep. | `run_md_stage`, `run_qmmm_stage` |
| `protocharge.generator.run_tc_resp` | Build and submit TeraChem RESP runs from configs. | `run_tc_resp` |
| `protocharge.generator.terachem_processing` | Process raw TeraChem outputs into standardized RESP inputs. | `process_terachem_outputs` |
| `protocharge.training.resp_parser` | Parse TeraChem `resp.out`, `esp.xyz`, and `.xyz` geometry frames. | `ParseRespDotOut`, `ParseESPXYZ`, `ParseDotXYZ` |
| `protocharge.training.linearESPcharges` | Build Coulomb matrices and solve the linear (unrestrained) ESP fit. | `prepare_linear_system`, `explicit_solution` |
| `protocharge.training.resp` | Single-frame RESP solver with hyperbolic restraint. | `fit_resp_charges`, `kkt_residual_at` |
| `protocharge.utils.dipole` | Dipole and center-of-mass helpers. | `center_of_mass_bohr_from_xyz` |
| `protocharge.training.symmetry` | WL refinement and symmetry buckets for charge tying. | `buckets_from_pdb` |
| `protocharge.training.twostepresp_basic` | Two-step RESP with total + bucket constraints. | `resp_step`, `solve_least_squares_with_constraints` |
| `protocharge.training.twostepresp_masked_total` | Two-step RESP with a masked total-charge constraint. | `load_total_constraint`, `build_total_constraint_mask` |
| `protocharge.training.twostepresp_group_constraints` | Two-step RESP with group constraints (no implicit total). | `load_group_constraints` |
| `protocharge.training.twostepresp_frozen_buckets` | Two-step RESP with frozen bucket values. | `load_frozen_buckets` |
| `protocharge.training.multiconfresp` | Multi-configuration assembly for a microstate. | `mcresp.py` CLI |
| `protocharge.training.reduced_basic` | Reduced-space RESP solver (total + bucket constraints). | `reduced.py` CLI |
| `protocharge.training.reduced_masked_total` | Reduced-space solver with masked total constraint. | `reduced.py` CLI |
| `protocharge.training.reduced_group_constraints` | Reduced-space solver with group constraints. | `reduced.py` CLI |
| `protocharge.training.multimoleculeresp` | Multi-molecule, multi-config RESP (global buckets + optional freezing). | `mmresp.py` CLI |
| `protocharge.validation.refep` | REFEP validation workflow (prep/run/grid/analyze). | `run_refep_stage` |
| `protocharge.validation.dipole` | Dipole validation against RESP/TeraChem outputs. | `run_dipole_validation` |

For recipes and command-line workflows, see the pages under **Workflows**.
