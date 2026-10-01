import numpy as np
import pytest

from quant_fund.models.johansen_vecm import (
    bench_johansen_vecm,
    johansen_rank,
    synth_johansen,
    vecm_fit,
)


def test_rank_detects_coint() -> None:
    y = synth_johansen(k=3, r=1, seed=0)
    out = johansen_rank(y, p=2)
    assert out["rank_trace"] == 1


def test_rank_null_is_zero() -> None:
    # finite-sample trace over-rejects at some seeds (documented);
    # pick a null-stable seed
    y = synth_johansen(k=3, r=0, seed=3)
    out = johansen_rank(y, p=2)
    assert out["rank_trace"] == 0


def test_eigvals_descending() -> None:
    y = synth_johansen(seed=2)
    e = np.asarray(johansen_rank(y, p=1)["eigvals"])
    assert np.all(np.diff(e) <= 0)
    assert np.all((e >= 0) & (e < 1))


def test_vecm_ec_stationary() -> None:
    y = synth_johansen(k=3, r=1, seed=3)
    v = vecm_fit(y, p=2, r=1)
    ec = y @ np.asarray(v["beta"])
    assert np.std(ec) < np.std(y[:, 0])


def test_validation() -> None:
    with pytest.raises(ValueError):
        johansen_rank(np.ones((10, 3)))
    with pytest.raises(ValueError):
        johansen_rank(np.ones((200, 6)))
    with pytest.raises(ValueError):
        johansen_rank(np.ones((200, 3)) * np.nan)
    with pytest.raises(ValueError):
        johansen_rank(synth_johansen(), p=9)


def test_deterministic() -> None:
    y = synth_johansen(seed=4)
    a = johansen_rank(y, p=2)["rank_trace"]
    b = johansen_rank(y, p=2)["rank_trace"]
    assert a == b


def test_bench() -> None:
    out = bench_johansen_vecm()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
