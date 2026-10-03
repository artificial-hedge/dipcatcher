"""joint_fit_bench — does one placement law close all four tape targets?"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.microstructure.joint_fit_bench import (
    JOINT_FIT_SCHEMA,
    _measure_cell,
    _score,
    joint_fit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def test_measure_cell_runs() -> None:
    m = _measure_cell(1.0, 20, 0.3, 0.0, 3000, 5)
    assert m["n_fills"] > 0
    assert m["g1_mean"] is not None and m["g1_mean"] >= 1.0
    assert m["instant_signed_ticks"] is not None
    assert set(m["kernel"]) == {"1", "5", "20", "50", "200"}


def test_score_normalized() -> None:
    cell = {
        "instant_signed_ticks": 0.887,
        "kernel": {"200": 4.6447},
        "g1_mean": 2.49,
        "spread_mean": 13.09,
    }
    assert _score(cell) == 0.0
    cell["instant_signed_ticks"] = None
    assert _score(cell) is None


def test_bench_seals(tmp_path: Path) -> None:
    out = joint_fit_bench(horizon=1500, seed=3)
    assert out["schema"] == JOINT_FIT_SCHEMA
    assert len(out["cells"]) == 16
    assert out["claims"]["best_cell"] is not None
    body = dict(out)
    sha = body.pop("receipt_sha256")
    assert hash_bytes(canonical_json_bytes(body)) == sha
    p = tmp_path / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(str(p))["valid"] is True
