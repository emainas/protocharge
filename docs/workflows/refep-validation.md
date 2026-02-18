# Validation: REFEP

This validation layer uses Hamiltonian replica exchange and REFEP energy grids to compare two endpoint charge sets (e.g., HID vs HIE). The workflow is YAML-driven and split into stages.

## Stages

- `refep.prep`: build endpoint topologies (tleap) and interpolate charges (parmed)
- `refep.run`: run H-REMD production
- `refep.grid`: demux + compute REFEP energy grid (sander)
- `refep.analyze`: analyze grid (BAR/TI, plots)

## CLI

```bash
pc --validate refep.prep --yaml configs/HID/refep.yaml
pc --validate refep.run --yaml configs/HID/refep.yaml --slurm
pc --validate refep.grid --yaml configs/HID/refep.yaml --slurm
pc --validate refep.analyze --yaml configs/HID/refep.yaml
```

## Config example

```yaml
microstate: HID
refep:
  prep:
    workdir: output/HID/refep/prep
    charge_exports:
      - charges:
          path: output/HID/twostepRESP_basic/charges.npz
          key: step2
          frame: -1
        mol2: configs/HID/hid.mol2
        resid_prefix: pept
        output: configs/HID/hid_charges.yaml
    tleap_inputs:
      - template: configs/HID/tleap_hid.in
        output: tleap_hid.in
        charges: configs/HID/hid_charges.yaml
      - template: configs/HID/tleap_hie.in
        output: tleap_hie.in
        charges: configs/HID/hie_charges.yaml
    files:
      - src: configs/HID/parmed.interpolate.in
        dst: parmed.interpolate.in
    commands:
      - ["tleap", "-f", "tleap_hid.in"]
      - ["tleap", "-f", "tleap_hie.in"]
      - ["parmed", "-i", "parmed.interpolate.in"]

  run:
    workdir: output/HID/refep/run
    files:
      - src: configs/HID/remd_hist.in
        dst: remd_hist.in
      - src: configs/HID/remd_hist.group
        dst: remd_hist.group
    commands:
      - ["pmemd.MPI", "-ng", "16", "-groupfile", "remd_hist.group"]
    slurm:
      partition: batch
      nodes: 4
      cpus_per_task: 64
      time: "24:00:00"

  grid:
    workdir: output/HID/refep/grid
    files:
      - src: configs/HID/demux_.in
        dst: demux_.in
      - src: configs/HID/refep_.in
        dst: refep_.in
    commands:
      - ["cpptraj", "-i", "demux_.in"]
      - ["bash", "run_refep_grid.sh"]
    slurm:
      partition: batch
      nodes: 4
      cpus_per_task: 64
      time: "04:00:00"

  analyze:
    workdir: output/HID/refep/analyze
    files:
      - src: configs/HID/plot_refep_du_grid.py
        dst: plot_refep_du_grid.py
    commands:
      - ["python", "plot_refep_du_grid.py"]
```

## Charge overrides

If `charges` is provided for a `tleap_inputs` entry, `protocharge` injects `set <res>.<atom> charge <q>` lines into the TLEaP template. By default it inserts after the marker line:

```
# BILIRESP_CHARGES
```

If the marker is missing, the block is appended at the end of the file.

## Charge export helper

`refep.prep` can export charge overrides directly from RESP outputs using `charge_exports`.

Supported charge files:

- `.npz` (choose `key`, default tries `step2`, `charges`, `q`)
- `.npy`

If the charge array is 2D, use `frame` (default `-1`) or `aggregate: mean`.

Mol2 parsing uses the `@<TRIPOS>ATOM` block. Residue labels are built from `subst_id` (default) or `subst_name` via `resid_source`. Use `resid_prefix` to generate `pept.<id>` style labels.
