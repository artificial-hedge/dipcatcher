from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.sign_predict import (
    _continuation,
    lobster_sign_predict,
    sign_predict_bench,
    sim_sign_predict,
)


def test_continuation_iid() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    signs = list(rng.choice([-1, 1], 5000))
    out = _continuation(signs)
    p5 = out["continuation"]["5"]["p_continue"]
    assert 0.35 < p5 < 0.65  # near 0.5 under iid


def test_continuation_perfect_runs() -> None:
    # alternating pairs: every run length is exactly 2
    signs = [s for _ in range(500) for s in (1, 1, -1, -1)]
    out = _continuation(signs)
    # after runlen>=2, next sign always flips
    assert out["continuation"]["2"]["p_continue"] == pytest.approx(0.0)


def test_too_few() -> None:
    assert _continuation([1, -1] * 10)["ok"] is False


def test_lobster_sign_predict_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        # 60 buy-aggressor execs in a row (resting sell, direction=-1)
        for i in range(61):
            wm.writerow([34200.0 + i, 4, i + 1, 10, 4000, -1])
            wo.writerow([4000, 100, 3000, 100])
    out = lobster_sign_predict(tmp_path)
    assert out["n_signs"] == 61
    # after long run of buys, continuation stays high here (all buys)
    assert out["continuation"]["1"]["p_continue"] == 1.0


def test_sim_predict_runs() -> None:
    out = sim_sign_predict(horizon=3000, seed=3)
    assert out["n_signs"] > 100


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sign_predict_bench(tmp_path)
