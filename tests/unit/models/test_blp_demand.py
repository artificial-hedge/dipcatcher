import numpy as np
import pytest

from quant_fund.models.blp_demand import (
    bench_blp_demand,
    blp_estimate,
    blp_invert,
    synth_blp_market,
)


def test_inversion_recovers_delta() -> None:
    d = synth_blp_market(sigma=0.5, seed=0)
    s = np.asarray(d["shares"])
    nu = np.random.default_rng(0).standard_normal(40)
    delta = blp_invert(s, np.asarray(d["x_rc"]), nu, 0.5, np.asarray(d["mkt"]))
    assert delta.shape == s.shape
    assert np.all(np.isfinite(delta))


def test_sigma_zero_logit_case() -> None:
    d = synth_blp_market(sigma=0.0, seed=1)
    s = np.asarray(d["shares"])
    nu = np.random.default_rng(0).standard_normal(40)
    delta = blp_invert(s, np.asarray(d["x_rc"]), nu, 0.0, np.asarray(d["mkt"]))
    # at σ=0 inversion is exact: δ_j = ln s_j − ln s0
    m = np.asarray(d["mkt"]) == 0
    s0 = 1.0 - float(np.sum(s[m]))
    expected = np.log(s[m]) - np.log(s0)
    assert np.allclose(delta[m], expected, atol=1e-6)


def test_alpha_recovery() -> None:
    d = synth_blp_market(alpha=1.0, sigma=1.0, seed=2)
    out = blp_estimate(
        np.asarray(d["shares"]),
        np.asarray(d["x"]),
        np.asarray(d["price"]),
        np.asarray(d["x_rc"]),
        np.asarray(d["z"]),
        np.asarray(d["mkt"]),
        n_draws=40,
        seed=3,
    )
    assert abs(float(out["alpha"]) - 1.0) < 0.2


def test_sigma_recovery() -> None:
    d = synth_blp_market(alpha=1.0, sigma=1.0, seed=4)
    out = blp_estimate(
        np.asarray(d["shares"]),
        np.asarray(d["x"]),
        np.asarray(d["price"]),
        np.asarray(d["x_rc"]),
        np.asarray(d["z"]),
        np.asarray(d["mkt"]),
        n_draws=60,
        seed=5,
    )
    assert abs(float(out["sigma"]) - 1.0) < 0.4


def test_synth_shapes() -> None:
    d = synth_blp_market(n_markets=5, n_products=8, seed=0)
    n = 5 * 8
    for key in ("shares", "price", "x_rc", "mkt"):
        assert np.asarray(d[key]).shape == (n,)
    assert np.all(np.asarray(d["shares"]) > 0)
    assert np.all(np.asarray(d["shares"]) < 1)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    nu = rng.standard_normal(30)
    with pytest.raises(ValueError):
        blp_invert(
            np.array([0.5, -0.1, 0.3]),
            rng.normal(0, 1, 3),
            nu,
            1.0,
            np.array([0.0, 0.0, 0.0]),
        )
    with pytest.raises(ValueError):
        blp_invert(
            np.array([0.4, 0.3, 0.3]),
            rng.normal(0, 1, 3),
            np.array([1.0, 2.0]),
            1.0,
            np.array([0.0, 0.0, 0.0]),
        )


def test_bench() -> None:
    out = bench_blp_demand()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
