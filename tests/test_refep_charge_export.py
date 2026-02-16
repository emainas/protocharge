from __future__ import annotations

from pathlib import Path

import numpy as np

from biliresp.validation.refep import run_refep_stage


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_charge_export_from_npz(tmp_path: Path) -> None:
    mol2 = tmp_path / "mol.mol2"
    _write(
        mol2,
        """@<TRIPOS>ATOM
1 C1 0 0 0 C 1 RES
2 H1 0 0 0 H 1 RES
@<TRIPOS>BOND
""",
    )
    charges = np.array([[0.1, 0.2], [-0.1, -0.2]])
    npz_path = tmp_path / "charges.npz"
    np.savez(npz_path, step2=charges)

    cfg = {
        "microstate": "HID",
        "refep": {
            "prep": {
                "workdir": str(tmp_path / "out"),
                "charge_exports": [
                    {
                        "charges": {"path": str(npz_path), "key": "step2", "frame": 1},
                        "mol2": str(mol2),
                        "resid_prefix": "pept",
                        "output": str(tmp_path / "charges.yaml"),
                    }
                ],
                "commands": [],
            }
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(cfg), encoding="utf-8")

    run_refep_stage("prep", cfg_path, dry_run=True)
    out = (tmp_path / "charges.yaml").read_text(encoding="utf-8")
    assert "pept.1" in out
    assert "C1" in out
    assert "0.2" in out
