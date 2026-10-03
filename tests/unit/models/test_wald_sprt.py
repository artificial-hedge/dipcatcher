import numpy as np
import pytest

from quant_fund.models.wald_sprt import (
    bench_wald_sprt,
    sprt_asn,
    sprt_bounds,
    sprt_normal,
    synth_sprt_stream,
)


def test_sprt_accepts_h1() -> None:
    rng = np.random.default_rng(0)
    out = sprt_normal(rng.normal(0.3, 1.0, 2000), 0.0, 0.3, 1.0)
    assert out["decision"] == 1.0


def test_sprt_rejects_h1_under_h0() -> None:
    rng = np.random.default_rng(1)
    out = sprt_normal(rng.normal(0.0, 1.0, 2000), 0.0, 0.3, 1.0)
    assert out["decision"] == 0.0


def test_bounds_signs() -> None:
    a, b = sprt_bounds(0.05, 0.05)
    assert a > 0 and b < 0


def test_asn_reasonable() -> None:
    asn = sprt_asn(0.3, 0.0, 0.3, 1.0)
    assert 20 < asn < 200


def test_stream_shape() -> None:
    s = synth_sprt_stream(n=50, seed=0)
    assert s.shape == (50,)


def test_validation() -> None:
    with pytest.raises(ValueError):
        sprt_bounds(0.9, 0.05)
    with pytest.raises(ValueError):
        sprt_normal(np.array([1.0, 2.0]), 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        sprt_normal(np.arange(10.0), 0.0, 1.0, -1.0)
    with pytest.raises(ValueError):
        sprt_normal(np.arange(10.0), 0.0, 0.0, 1.0)


def test_bench() -> None:
    out = bench_wald_sprt()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
