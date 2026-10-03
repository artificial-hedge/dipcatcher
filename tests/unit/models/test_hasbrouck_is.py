"""Tests for hasbrouck_is — information share price discovery."""

import numpy as np
import pytest

from quant_fund.models.hasbrouck_is import (
    bench_hasbrouck,
    information_share,
    synth_hasbrouck,
)


def test_informed_venue_dominates() -> None:
    ci, _ = synth_hasbrouck(seed=1)
    r = information_share(ci)
    assert r["is1_lo"] > 0.6
    assert r["gg_w1"] > 0.7


def test_symmetric_null_splits() -> None:
    _, null = synth_hasbrouck(seed=2)
    r = information_share(null)
    assert 0.3 < r["is1_mid"] < 0.7


def test_bounds_bracket_mid() -> None:
    ci, _ = synth_hasbrouck(seed=3)
    r = information_share(ci)
    assert r["is1_lo"] <= r["is1_mid"] <= r["is1_hi"]


def test_beta_recovers_cointegration() -> None:
    ci, _ = synth_hasbrouck(seed=4)
    r = information_share(ci)
    assert 0.8 < r["beta"] < 1.2


def test_fail_closed() -> None:
    ci, _ = synth_hasbrouck(seed=5)
    with pytest.raises(ValueError):
        information_share(ci[:50])
    with pytest.raises(ValueError):
        information_share(np.full((500, 2), np.nan))
    with pytest.raises(ValueError):
        information_share(np.ones((500, 3)))


def test_determinism() -> None:
    ci, _ = synth_hasbrouck(seed=6)
    assert information_share(ci) == information_share(ci)


def test_bench_schema_and_score() -> None:
    r = bench_hasbrouck()
    for k in ("is1_mid", "is1_lo", "is1_null_mid", "gg_w1", "beta", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
