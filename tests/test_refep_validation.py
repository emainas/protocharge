from __future__ import annotations

from pathlib import Path

import yaml

from protocharge.validation.refep import run_refep_stage


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_refep_prep_inserts_charges(tmp_path: Path) -> None:
    cfg_path = tmp_path / "configs" / "refep.yaml"
    tleap_template = tmp_path / "tleap_hid.in"
    charges_path = tmp_path / "hid_charges.yaml"

    _write(
        tleap_template,
        "source leaprc.constph\n# BILIRESP_CHARGES\nsaveamberparm x hid.parm7 hid.rst7\nquit\n",
    )
    _write(
        charges_path,
        yaml.safe_dump([{"resid": "pept.2", "atom": "N", "charge": -0.123}]),
    )

    cfg = {
        "microstate": "HID",
        "refep": {
            "prep": {
                "workdir": str(tmp_path / "out"),
                "tleap_inputs": [
                    {
                        "template": str(tleap_template),
                        "output": "tleap_hid.in",
                        "charges": str(charges_path),
                    }
                ],
                "commands": [],
            }
        },
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = run_refep_stage("prep", cfg_path, dry_run=True)
    rendered = (out_dir / "tleap_hid.in").read_text(encoding="utf-8")
    assert "set pept.2.N charge -0.123" in rendered
