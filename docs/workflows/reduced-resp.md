# Reduced-Space RESP

Reduced-space solvers enforce the linear constraints exactly by projecting into the nullspace of the constraint matrix. These are useful for large ensemble fits where a direct KKT solve is unstable.

## Variants

- **Basic** – total + bucket constraints: `protocharge.training.reduced_basic`
- **Masked total** – total charge applied to a subset: `protocharge.training.reduced_masked_total`
- **Group constraints** – arbitrary group targets: `protocharge.training.reduced_group_constraints`

## Example

```bash
python -m protocharge.training.reduced_group_constraints.reduced \
  --microstate HID \
  --load-and-resp \
  --input-dir-name multiconfRESP \
  --output-dir-name multiconfRESP_reduced_group_constraints
```

The outputs are written under `output/<microstate>/` in the selected output directory.
