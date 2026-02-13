# Raw TeraChem Processing

Most users should run TeraChem RESP jobs with `biliresp --run-tc-resp`, then collect outputs with:

```bash
biliresp --process-tc-resp data/microstates/<MICROSTATE>
```

This gathers `resp.out` plus the scratch `esp.xyz` files and writes the standardized inputs under `data/microstates/<MICROSTATE>/terachem/`.

## Required raw layout

Place raw outputs under:

```
data/microstates/<MICROSTATE>/raw_terachem_outputs/
```

Expected structure:

```
raw_terachem_outputs/
  confs/
    conf_001/
      resp.out
      scr.<rst7_file_name>/esp.xyz
    conf_002/
      resp.out
      scr.<rst7_file_name>/esp.xyz
```

## Outputs

The command writes:

```
data/microstates/<MICROSTATE>/terachem/respout/conf####.resp.out
data/microstates/<MICROSTATE>/terachem/espxyz/conf####.esp.xyz
```

## Next step

Once processed, proceed to the RESP workflows (two-step, multiconf, reduced, etc.) using the standard configs under `configs/`.
