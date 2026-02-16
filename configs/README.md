# Configs

This folder holds YAML configuration files for `protocharge` runs.

Recommended layout:

```
configs/<microstate>/<function>/config.yaml
```

Examples:

```
configs/HID/twostepRESP_basic/config.yaml
configs/HIP/multiconfRESP/config.yaml
configs/PRN_prot/multiconfRESP_reduced_masked_total/config.yaml
```

Each config is a YAML mapping with (at minimum):

```
microstate: HID
args:
  # arguments passed to the underlying module/script
  maxiter: 400
  frame: -1
slurm:
  # optional Slurm overrides
  time: "24:00:00"
  mem: "64G"
```

Relative paths inside `args:` are resolved relative to the config file location.
