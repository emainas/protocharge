# TeraChem RESP Job Submission

This workflow writes TeraChem input files (`resp.in`), generates Slurm submission scripts, and submits jobs for each configuration under a microstate.

## Expected inputs

```
input/microstates/<MICROSTATE>/input_tc_structures/confs/<conf>/parm7
input/microstates/<MICROSTATE>/input_tc_structures/confs/<conf>/rst7
configs/<MICROSTATE>/input_tc_structures/<conf>/config.yaml
```

Each `config.yaml` provides the TeraChem settings, including the charge:

```yaml
charge: 1
basis: 6-31gs
method: rhf
spinmult: 1
maxit: 1000
run: energy
resp: yes
esp_grid_dens: 4.0
```

## Run

```bash
pc --run-tc-resp input/microstates/<MICROSTATE>
```

For each conf, the module writes:

- `resp.in`
- `run_tc_resp.slurm`
- `nowater.parm7` / `nowater.rst7` (copies of the provided files)

Then it submits `sbatch run_tc_resp.slurm` immediately.

## Collect outputs for RESP workflows

TeraChem writes `resp.out` in the conf directory and `esp.xyz` under a scratch folder named `scr.<rst7>`. Once the jobs finish, collect outputs into the standard layout:

```bash
pc --process-tc-resp input/microstates/<MICROSTATE>
```

This writes:

```
input/microstates/<MICROSTATE>/terachem/respout/conf####.resp.out
input/microstates/<MICROSTATE>/terachem/espxyz/conf####.esp.xyz
```
