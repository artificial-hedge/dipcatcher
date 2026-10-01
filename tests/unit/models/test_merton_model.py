import numpy as np
import pytest

from quant_fund.models.merton_model import (
    bench_merton_model,
    merton_equity,
    merton_invert,
    synth_merton,
)


def test_merton_equity_is_call() -> None:
    e = merton_equity(130.0, 100.0, 0.25, 0.02, 1.0)
    assert 30.0 < e < 35.0
    e_low = merton_equity(105.0, 100.0, 0.4, 0.02, 1.0)
    assert 0 < e_low < e


def test_merton_invert_recovers_v() -> None:
    d = synth_merton(v=130.0, sigma_v=0.25, seed=2)
    out = merton_invert(np.asarray(d["equity"]), float(d["equity_vol"]), float(d["debt"]))
    assert abs(out["asset_value"] / float(d["v_true"]) - 1) < 0.2


def test_merton_dd_ordering() -> None:
    safe = synth_merton(v=130.0, sigma_v=0.25, seed=3)
    lev = synth_merton(v=105.0, sigma_v=0.4, seed=4)
    a = merton_invert(np.asarray(safe["equity"]), float(safe["equity_vol"]), 100.0)
    b = merton_invert(np.asarray(lev["equity"]), float(lev["equity_vol"]), 100.0)
    assert a["dd"] > b["dd"]
    assert a["default_prob"] < b["default_prob"]


def test_merton_validation() -> None:
    with pytest.raises(ValueError):
        merton_equity(-1.0, 100.0, 0.25, 0.02, 1.0)
    with pytest.raises(ValueError):
        merton_invert(np.array([-5.0]), 0.3, 100.0)
    with pytest.raises(ValueError):
        merton_invert(np.array([50.0]), -0.1, 100.0)


def test_merton_deterministic() -> None:
    d = synth_merton(seed=5)
    a = merton_invert(np.asarray(d["equity"]), float(d["equity_vol"]), 100.0)
    b = merton_invert(np.asarray(d["equity"]), float(d["equity_vol"]), 100.0)
    assert a["dd"] == b["dd"]


def test_bench_merton() -> None:
    out = bench_merton_model()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
