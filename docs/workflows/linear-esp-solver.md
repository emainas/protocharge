# Linear ESP Charge Solver

The linear solver fits electrostatic potential (ESP) charges that reproduce grid values exported by RESP/TeraChem. Everything lives in `biliresp.linearESPcharges.linear` and is backed by numpy.

## Pipeline

1. **Parse Terachem RESP output** with `ParseRespDotOut` (from `biliresp.resp_parser`) to obtain atomic positions and RESP ESP charges for each frame.
2. **Read ESP grid** points from `esp.xyz`.
3. **Build the design matrix** `A` where `A[i, j] = 1 / r_ij` for grid point `i` and atom `j`.
4. **Solve** the constrained optimization problem `A q ≈ V` subject to `Σ q = Q` with `explicit_solution`, a closed-form projection that first finds the unconstrained least squares solution and then enforces the total charge (Lagrange multiplier method). It accepts an optional `ridge` hyper-parameter if you want to add a small diagonal Tikhonov term for numerical stability.

## Script entry point

The `scripts/compare_charges.py` wrapper prepares the system and prints per-atom differences between RESP unrestrained charges and the fitted charges:

```bash
python scripts/compare_charges.py data/raw/resp.out data/raw/esp.xyz 78 --frame -1
```

- `resp.out`: RESP log file containing the ESP unrestrained block.
- `esp.xyz`: grid potentials from TeraChem.
- `78`: number of atoms in the RESP job.
- `--frame`: zero-based frame index (use `-1` for the last frame).
- The output lists each atom index with the RESP charge, the fitted charge, and the difference. A footer prints charge conservation, RMSE, and RRMS metrics so you can quickly gauge the fit quality.

## Mathematical derivation

The objective is to minimise the grid misfit while enforcing the desired total charge:

$$
\begin{aligned}
\min_{q}\ & \frac{1}{2}\lVert A q - V \rVert_2^2 \\
\text{s.t.}\ & \mathbf{1}^\top q = Q.
\end{aligned}
$$

Using the Lagrangian $\mathcal{L}(q, \lambda) = \tfrac{1}{2}\lVert A q - V \rVert_2^2 + \lambda (\mathbf{1}^\top q - Q)$ gives the stationary conditions

$$
\nabla_q \mathcal{L} = A^\top(Aq - V) + \lambda\,\mathbf{1} = 0,\qquad \nabla_\lambda \mathcal{L} = \mathbf{1}^\top q - Q = 0.
$$

Defining $H = A^\top A$ and $g = A^\top V$ leads to the Karush–Kuhn–Tucker system

$$
\begin{pmatrix}
H & \mathbf{1} \\
\mathbf{1}^\top & 0
\end{pmatrix}
\begin{pmatrix}
q \\
\lambda
\end{pmatrix} =
\begin{pmatrix}
g \\
Q
\end{pmatrix}.
$$

Rather than solving the $(N+1)\times(N+1)$ block system directly, the implementation projects the unconstrained least-squares solution onto the charge-conserving hyperplane:

1. Solve $H q_0 = g$ to obtain the unconstrained solution (`q0`).
2. Solve $H c = \mathbf{1}$ for the correction direction (`c`).
3. Compute $\alpha = \mathbf{1}^\top c$ and $s = \mathbf{1}^\top q_0$.
4. Project: $q = q_0 - \frac{s - Q}{\alpha}\, c$.

If numerical damping is requested, $H$ is replaced with $H + \eta I$ (with a small $\eta > 0$) in the two linear solves above. The procedure yields the same result as the full KKT solve but avoids explicitly forming the block matrix.

## Programmatic use

You can access the components directly:

```python
from biliresp.linearESPcharges.linear import prepare_linear_system, explicit_solution

A, V, Q, resp_charges = prepare_linear_system("data/raw/resp.out", "data/raw/esp.xyz", 78, frame_index=-1)
solver = explicit_solution(ridge=0.0)
result = solver.fit(A, V, Q)
print(result["rmse"], result["sum_q"])
```

The returned dictionary includes the fitted charges `q`, intermediate matrices, RMSE/RRMS, and the enforced total charge.
