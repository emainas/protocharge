from __future__ import annotations

import argparse
from pathlib import Path

from biliresp.validation.dipole import run_dipole_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print QM, ESP-unrestrained, and fitted dipole moments for the selected frame."
    )
    parser.add_argument("resp_out", type=Path)
    parser.add_argument("esp_xyz", type=Path)
    parser.add_argument("geom_xyz", type=Path)
    parser.add_argument("n_atoms", type=int)
    parser.add_argument("--frame", type=int, default=-1, help="Frame index (default: last)")
    args = parser.parse_args()

    tmp_cfg = {
        "resp_out": str(args.resp_out),
        "esp_xyz": str(args.esp_xyz),
        "geom_xyz": str(args.geom_xyz),
        "n_atoms": args.n_atoms,
        "frame": args.frame,
    }
    cfg_path = Path(".dipole_validation_tmp.yaml")
    cfg_path.write_text(__import__("yaml").safe_dump(tmp_cfg), encoding="utf-8")
    dipoles = run_dipole_validation(cfg_path)
    cfg_path.unlink(missing_ok=True)

    print("QM (from resp.out log)")
    print("  vector (Debye):", dipoles["qm_dipole_vec_D"])
    print("  |μ| (Debye):   {:.6f}".format(dipoles["qm_dipole_mag_D"]))
    print()
    print("Terachem charges (RESP log)")
    print("  vector (Debye):", dipoles["terachem_dipole_vec_D"])
    print("  |μ| (Debye):   {:.6f}".format(dipoles["terachem_dipole_mag_D"]))
    print("  Δ vector vs QM (Debye):", dipoles["delta_terachem_vs_qm_vec_D"])
    print("  Δ|μ| vs QM (Debye):     {:.6f}".format(dipoles["delta_terachem_vs_qm_mag_D"]))
    print()
    print("Lagrange multiplier fit (explicit)")
    print("  vector (Debye):", dipoles["lagrange_dipole_vec_D"])
    print("  |μ| (Debye):   {:.6f}".format(dipoles["lagrange_dipole_mag_D"]))
    print("  Δ vector vs QM (Debye):", dipoles["delta_lagrange_vs_qm_vec_D"])
    print("  Δ|μ| vs QM (Debye):     {:.6f}".format(dipoles["delta_lagrange_vs_qm_mag_D"]))

if __name__ == "__main__":
    main()
