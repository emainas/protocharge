from pathlib import Path
import yaml

from protocharge.training.twostepresp_frozen_buckets.tsresp import load_frozen_buckets


def test_load_frozen_buckets(tmp_path: Path):
    data = [
        {"bucket": 0, "value": 0.1},
        {"bucket": 2, "value": -0.3},
    ]
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    frozen = load_frozen_buckets(path)
    assert frozen == {0: 0.1, 2: -0.3}
