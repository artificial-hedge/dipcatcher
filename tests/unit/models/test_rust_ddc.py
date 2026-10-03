import numpy as np
import pytest

from quant_fund.models.rust_ddc import (
    bench_rust_ddc,
    rust_ccp,
    rust_nfxp,
    rust_value,
    synth_rust,
)


def test_value_converges_and_monotone() -> None:
    d = synth_rust(seed=0)
    ev = rust_value(0.1, 8.0, np.asarray(d["trans"]))
    # keeping is worse at high wear: EV declines in x
    assert ev[0] > ev[-1]


def test_ccp_increases_with_wear() -> None:
    d = synth_rust(seed=0)
    p = rust_ccp(0.1, 8.0, np.asarray(d["trans"]))
    assert p[-1] > p[0]
    assert np.all(p > 0) and np.all(p < 1)


def test_ccp_higher_when_replacement_cheap() -> None:
    d = synth_rust(seed=1)
    p_cheap = rust_ccp(0.1, 3.0, np.asarray(d["trans"]))
    p_dear = rust_ccp(0.1, 20.0, np.asarray(d["trans"]))
    assert np.all(p_cheap >= p_dear)


def test_nfxp_recovers() -> None:
    d = synth_rust(theta=0.1, rc=8.0, seed=2)
    est = rust_nfxp(np.asarray(d["x"]), np.asarray(d["choice"]), np.asarray(d["trans"]))
    assert abs(est["rc"] - 8.0) < 3.0
    assert abs(est["theta"] - 0.1) < 0.08


def test_synth_shapes() -> None:
    d = synth_rust(n_periods=200, seed=3)
    assert np.asarray(d["x"]).shape == (200,)
    assert np.asarray(d["trans"]).shape == (30, 30)
    row_sums = np.asarray(d["trans"]).sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    tr = np.eye(10)
    with pytest.raises(ValueError):
        rust_value(0.1, 8.0, np.eye(3))
    with pytest.raises(ValueError):
        rust_value(-0.1, 8.0, tr)
    with pytest.raises(ValueError):
        rust_value(0.1, -8.0, tr)
    with pytest.raises(ValueError):
        rust_nfxp(rng.uniform(0, 5, 20), rng.binomial(1, 0.5, 20).astype(float), tr)
    with pytest.raises(ValueError):
        rust_nfxp(
            rng.uniform(0, 5, 100),
            np.full(100, 2.0),
            tr,
        )
    with pytest.raises(ValueError):
        rust_nfxp(
            rng.uniform(0, 99, 100),
            rng.binomial(1, 0.5, 100).astype(float),
            tr,
        )


def test_bench() -> None:
    out = bench_rust_ddc()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
