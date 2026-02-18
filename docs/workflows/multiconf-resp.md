# Multiconfiguration RESP (Ensemble)

The multiconfiguration workflow stacks multiple conformations of a single microstate into one large ESP fitting problem. It is useful when you want charges that best represent an ensemble rather than a single conformer.

## Assemble and Run

The CLI lives in `protocharge.multiconfresp.mcresp`.

```bash
python -m protocharge.multiconfresp.mcresp \
  --microstate PPP \
  --load-and-resp \
  --maxiter 400
```

Under the hood this:

- Reads `input/microstates/<microstate>/terachem/respout/*.resp.out` and `espxyz/*.esp.xyz`.
- Stacks the Coulomb matrices and ESP vectors across all configurations.
- Runs the two-step RESP solver (basic variant) over the stacked system.

## Output

For ensemble runs the default outputs are written under:

```
output/<microstate>/multiconfRESP/
```

If you are using a reduced-space variant, see **Reduced RESP** for the alternative output directories.
