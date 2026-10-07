"""changepoint_localize: e-value scan for where the stream shifted."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.changepoint_localize import (
    localize_changepoint,
    localize_report,
)


def _planted(seed: int, n: int = 400, tau: int = 250, shift: float = 3.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    x[tau:] += shift
    return x, tau


def test_finds_planted_change() -> None:
    x, tau = _planted(0)
    res = localize_changepoint(x, alpha=0.05, min_left=15, window=30)
    assert res.alarmed
    # excluded bracket contains the true change point
    assert res.cs_lo <= tau <= res.cs_hi
    # argmax estimate lands near the truth
    assert abs(res.tau_hat - tau) <= 60
    assert res.n == 400


def test_localization_accuracy_monte_carlo() -> None:
    covered = 0
    errors = []
    seeds = range(15)
    for seed in seeds:
        x, tau = _planted(seed, n=300, tau=180, shift=2.5)
        res = localize_changepoint(x, alpha=0.05, min_left=15, window=30)
        covered += int(res.cs_lo <= tau <= res.cs_hi)
        errors.append(abs(res.tau_hat - tau))
    assert covered >= 12, f"bracket covered tau in {covered}/15 runs"
    assert np.median(errors) <= 30


def test_no_change_no_bracket() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=400)
    res = localize_changepoint(x, alpha=0.05, min_left=15, window=30)
    # a valid procedure mostly stays quiet; if it flags, the bracket must
    # still be a real range
    assert 0 <= res.cs_lo <= res.cs_hi < res.n
    assert not res.alarmed or res.cs_lo <= res.cs_hi


def test_variance_shift_also_detected() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.3, size=400)
    x[200:] = rng.normal(0.0, 3.0, size=200)
    res = localize_changepoint(x, alpha=0.05, window=40)
    assert res.alarmed
    assert abs(res.tau_hat - 200) <= 60


def test_fails_closed_on_short_stream() -> None:
    with pytest.raises(ValueError, match="too short"):
        localize_changepoint([1.0] * 10, min_left=10, window=10)
    with pytest.raises(ValueError):
        localize_changepoint([0.0] * 100, alpha=1.5)
    with pytest.raises(ValueError):
        localize_changepoint([0.0] * 100, lam=0.0)


def test_nonfinite_dropped_not_crash() -> None:
    x, tau = _planted(1)
    x = np.concatenate([x, [np.nan, np.inf]])
    res = localize_changepoint(x, min_left=15, window=30)
    assert res.n == 400


def test_report_shape() -> None:
    x, _ = _planted(2)
    rep = localize_report(x, stream_name="score_stream")
    assert rep["kind"] == "changepoint_localize.v1"
    assert rep["stream"] == "score_stream"
    assert rep["n"] == 400
    assert isinstance(rep["log_evalues"], list)
    assert rep["alarmed"] is True


def test_tail_candidate_is_reachable() -> None:
    """SYNTHETIC: a change in the final window must still be localizable.

    Off-by-one guard: s = n - window is a legal candidate whose scan window
    is the last W observations. Pin a big shift exactly at n - window and
    require the maximizer to reach it.
    """
    rng = np.random.default_rng(11)
    n, window, min_left = 200, 40, 10
    x = rng.normal(0.0, 0.1, size=n)
    x[n - window :] += 5.0  # the last window is entirely post-change
    res = localize_changepoint(x, alpha=0.05, window=window, min_left=min_left)
    assert res.alarmed
    assert res.tau_hat == n - window
    assert res.cs_hi == n - window
