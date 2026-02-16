# Generator: MD + QMMM Training Set

This workflow generates the RESP training set from classical MD followed by QMMM and RESP/ESP evaluation. MD and QMMM are separate stages.

## CLI

```bash
biliresp --generator md.prep --yaml configs/HID/training.yaml
biliresp --generator md.run --yaml configs/HID/training.yaml --slurm

biliresp --generator qmmm.prep --yaml configs/HID/training.yaml
biliresp --generator qmmm.run --yaml configs/HID/training.yaml --slurm
biliresp --generator qmmm.esp --yaml configs/HID/training.yaml --slurm
```

## Config example

```yaml
microstate: HID

md:
  prep:
    workdir: results/HID/md/prep
    tleap_inputs:
      - template: configs/HID/tleap.in
        output: tleap.in
        charges: configs/HID/hid_charges.yaml
    commands:
      - [\"tleap\", \"-f\", \"tleap.in\"]
  run:
    workdir: results/HID/md/run
    commands:
      - [\"bash\", \"run.sh\"]
    slurm:
      partition: batch
      nodes: 1
      cpus_per_task: 16
      time: \"24:00:00\"

qmmm:
  prep:
    workdir: results/HID/qmmm
    parm7: results/HID/md/run/hid.parm7
    traj: results/HID/md/run/prod.nc
    frames:
      start: 1000
      end: 20000
      step: 100
    ambermask: \"339\"
    closest_n: 200
    region_range: \"0-50\"
    charge: 0
    tc:
      basis: def2-svp
      method: wb97xd3
      dftd: d3
      spinmult: 1
      maxit: 1000
      run: md
      nstep: 20000
      timestep: 0.5
      mdbc: spherical
      wmodel: tip3p

  run:
    workdir: results/HID/qmmm

  esp:
    workdir: results/HID/qmmm
    frames:
      start: 1000
      end: 20000
      step: 100
    charge: 0
    tc:
      basis: def2-svp
      method: wb97xd3
      dftd: d3
      spinmult: 1
      maxit: 1000
      esp_grid_dens: 4.0
```

## Outputs

- MD outputs under `results/<microstate>/md/`
- QMMM frame prep and jobs under `results/<microstate>/qmmm/`
- QMMM RESP outputs can be collected with:

```bash
biliresp --process data/microstates/<MICROSTATE>
```
