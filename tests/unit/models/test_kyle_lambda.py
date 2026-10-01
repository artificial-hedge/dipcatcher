import numpy as np
import pytest

from quant_fund.models.kyle_lambda import (
    bench_kyle_lambda,
    kyle_equilibrium,
    kyle_lambda,
    synth_kyle,
)


def test_equilibrium_identities() -> None:
    eq = kyle_equilibrium(4.0, 2.0)
    assert eq["lambda"] == pytest.approx(0.25)
    assert eq["beta"] == pytest.approx(2.0)
    assert eq["info_share"] == pytest.approx(0.5)


def test_lambda_recovers_equilibrium() -> None:
    eq = kyle_equilibrium(4.0, 2.0)
    d = synth_kyle(sigma_u=4.0, sigma_v=2.0, sigma_eps=0.3, seed=0)
    out = kyle_lambda(d["signed_flow"], d["price_change"])
    assert abs(out["lambda"] / eq["lambda"] - 1) < 0.15
    assert out["p_lambda"] < 0.01


def test_zero_flow_signal() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    dp = rng.normal(0, 1, 300)
    out = kyle_lambda(y, dp)
    assert abs(out["lambda"]) < 0.5


def test_validation() -> None:
    with pytest.raises(ValueError):
        kyle_lambda(np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        kyle_lambda(np.ones(50), np.ones(49))
    with pytest.raises(ValueError):
        kyle_lambda(np.ones(60) * np.nan, np.ones(60))
    with pytest.raises(ValueError):
        kyle_equilibrium(-1.0, 2.0)


def test_deterministic() -> None:
    d = synth_kyle(seed=2)
    a = kyle_lambda(d["signed_flow"], d["price_change"])
    b = kyle_lambda(d["signed_flow"], d["price_change"])
    assert a["lambda"] == b["lambda"]


def test_bench() -> None:
    out = bench_kyle_lambda()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
