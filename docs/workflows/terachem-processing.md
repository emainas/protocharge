# Raw TeraChem Processing

Most users will start with raw TeraChem output. Before running any RESP workflow, the raw outputs must be organized into the expected layout under `data/microstates/<MICROSTATE>/terachem/`.

This command collects the raw files and writes the standardized inputs:

```bash
biliresp --process data/microstates/<MICROSTATE>
```

## Required raw layout

Place raw outputs under:

```
data/microstates/<MICROSTATE>/raw_terachem_outputs/
```

Expected structure (mirrors the original TeraChem output layout):

```
raw_terachem_outputs/
  frame1/
    scr.carved_frame1/
      splits/
        conf1/
          resp-vac.out
          scr.nowater/esp.xyz
        conf2/
          resp-vac.out
          scr.nowater/esp.xyz
  frame2/
    scr.carved_frame2/
      splits/
        ...
```

## Outputs

The command writes:

```
data/microstates/<MICROSTATE>/terachem/respout/conf####.resp.out
data/microstates/<MICROSTATE>/terachem/espxyz/conf####.esp.xyz
```

## Custom raw subdirectory

If your raw outputs live in a different folder name:

```bash
biliresp --process data/microstates/<MICROSTATE> --tc-raw-subdir raw_tc
```

## Next step

Once processed, proceed to the RESP workflows (two-step, multiconf, reduced, etc.) using the standard configs under `configs/`.
