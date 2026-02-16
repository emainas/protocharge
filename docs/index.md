# protocharge

Utilities for analyzing electrostatic potential (ESP) outputs. The project includes a RESP parser for TeraChem, linear and hyperbolic solvers, two-step RESP workflows, and multi-configuration/multi-molecule pipelines.

Use the navigation to find quick-start installation instructions and focused walkthroughs for the workflows:

- Fitting charges with the linear ESP solver.
- Two-step RESP with symmetry and charge constraints.
- Multi-configuration and multi-molecule RESP.

Project layout:

- `data/` holds inputs and parameters tied to a microstate (PDB, esp.xyz, resp.out, symmetry buckets).
- `configs/` holds YAML run configurations and charge-constraint files.
- `results/` holds outputs organized by microstate and function.
