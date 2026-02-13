from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from biliresp.linearESPcharges import explicit_solution, prepare_linear_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fitted charges against RESP unrestrained charges.")
    parser.add_argument("resp_out", type=Path)
    parser.add_argument("esp_xyz", type=Path)
    parser.add_argument("n_atoms", type=int)
    parser.add_argument("--frame", type=int, default=-1, help="Frame index (default: last)")
    args = parser.parse_args()

    A, V, Q, resp_charges = prepare_linear_system(
        args.resp_out,
        args.esp_xyz,
        args.n_atoms,
        frame_index=args.frame,
    )

    solver = explicit_solution()
    res = solver.fit(A, V, Q)

    diff = res["q"] - resp_charges
    for idx, (resp_q, fitted_q, delta) in enumerate(zip(resp_charges, res["q"], diff)):
        print(
            f"{idx:4d}  resp={resp_q:+.6f}  fit={fitted_q:+.6f}  diff={delta:+.6e}"
        )

    print(
        f"\nΣq={res['sum_q']:.12f}, RMSE={res['rmse']:.6e}, RRMS={res['rrms']:.6e}"
    )


if __name__ == "__main__":
    main()
