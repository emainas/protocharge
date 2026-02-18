# Generator: MD + QMMM Training Set

This workflow generates the RESP training set from classical MD followed by QMMM and RESP/ESP evaluation. MD and QMMM are separate stages.

## CLI

```bash
pc --generator md.prep --yaml configs/HID/generator.yaml
pc --generator md.run --yaml configs/HID/generator.yaml --slurm

pc --generator qmmm.prep --yaml configs/HID/generator.yaml
pc --generator qmmm.run --yaml configs/HID/generator.yaml --slurm
pc --generator qmmm.esp --yaml configs/HID/generator.yaml --slurm
```

## Config example

```yaml
microstate: HID

md:
  prep:
    workdir: output/HID/md/prep
    tleap:
      module: "amber/25-CUDA_12.6-GCC12.2.0"
      binary: "tleap"
      mol2: input/microstates/HID/initial-files/ligand.mol2
      frcmod: input/microstates/HID/initial-files/ligand.frcmod
      frcmod_ion: frcmod.ionsjc_tip3p
      leaprc_mol: leaprc.gaff2
      leaprc_sol: leaprc.water.tip3p
      water_model: TIP3PBOX
      buffer: 15.0
      counterion: Na+
      counterion_num: 0
      prefix: HID
  run:
    workdir: output/HID/md/run
    inputs:
      parm7: output/HID/md/prep/HID.parm7
      rst7: output/HID/md/prep/HID.rst7
    runtime:
      module: "amber/25-CUDA_12.6-GCC12.2.0"
      executable: "pmemd.cuda"
      env: {}
      strict_mode: true
    stages:
      - name: min
        description: Minimization
        cntrl:
          imin: 1
          maxcyc: 2000
          ncyc: 1000
          cut: 10.0
        traj: false
      - name: heat
        description: Heating
        cntrl:
          imin: 0
          nstlim: 50000
          dt: 0.002
          tempi: 0.0
          temp0: 300.0
          ntb: 1
          ntt: 3
          gamma_ln: 2.0
          cut: 10.0
        wt:
          - type: TEMP0
            istep1: 0
            istep2: 50000
            value1: 0.0
            value2: 300.0
          - type: END
        traj: true
      - name: equil-nvt
        description: Equilibration NVT
        cntrl:
          imin: 0
          nstlim: 50000
          dt: 0.002
          temp0: 300.0
          ntb: 1
          ntt: 3
          gamma_ln: 2.0
          cut: 10.0
        traj: true
      - name: equil-npt
        description: Equilibration NPT
        cntrl:
          imin: 0
          nstlim: 50000
          dt: 0.002
          temp0: 300.0
          ntb: 2
          ntp: 1
          pres0: 1.0
          taup: 2.0
          ntt: 3
          gamma_ln: 2.0
          cut: 10.0
        traj: true
    slurm:
      profile: longleaf_gpu
      profiles:
        longleaf_gpu:
          partition: l40-gpu
          qos: gpu_access
          gres: gpu:1
          time: "24:00:00"
          nodes: 1
          cpus_per_task: 16
        sycamore_cpu:
          partition: batch
          time: "24:00:00"
          nodes: 1
          cpus_per_task: 16
      job_name: HID-md
      output: "slurm-%j.out"
      error: "slurm-%j.err"

qmmm:
  prep:
    workdir: output/HID/qmmm
    parm7: output/HID/md/run/hid.parm7
    traj: output/HID/md/run/prod.nc
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
    workdir: output/HID/qmmm

  esp:
    workdir: output/HID/qmmm
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

- MD outputs under `output/<microstate>/md/`
- QMMM frame prep and jobs under `output/<microstate>/qmmm/`
- QMMM RESP outputs can be collected with:

```bash
pc --process input/microstates/<MICROSTATE>
```
