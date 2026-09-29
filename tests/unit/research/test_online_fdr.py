"""online_fdr: Foster-Stine alpha-investing over a p-value stream."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.online_fdr import OnlineFDR, _gamma


def test_gamma_sums_to_one() -> None:
    assert _gamma(0) == pytest.approx(6.0 / np.pi**2)
    assert sum(_gamma(j) for j in range(10000)) == pytest.approx(1.0, abs=1e-4)


def test_null_stream_no_rejections() -> None:
    """Under the global null (uniform p) rejections stay rare at level 0.05."""
    total = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        proc = OnlineFDR(level=0.05)
        for p in rng.uniform(0, 1, size=200):
            proc.update(float(p))
        total += len(proc.rejections)
    # Each stream at mFDR<=0.05 allows ~2.5% expected false rate; 40 streams
    # x 200 tests, observed rejections should stay well under 10%.
    assert total <= 40


def test_signal_stream_rejects_and_wealth_regenerates() -> None:
    rng = np.random.default_rng(1)
    proc = OnlineFDR(level=0.05)
    ps = np.concatenate([rng.beta(0.5, 50, size=60), rng.uniform(0, 1, size=40)])
    for p in ps:
        proc.update(float(p))
    rep = proc.stream_report()
    assert rep["n_rejections"] >= 30
    # Wealth regenerates on rejections: post-signal wealth > spent initial.
    assert rep["final_wealth"] > 0.0


def test_dry_streak_spends_less() -> None:
    proc = OnlineFDR(level=0.05)
    wagers = []
    for _ in range(20):
        st = proc.update(0.99)
        wagers.append(st.alpha_t)
    # Gamma decay on a dry streak: each wager strictly smaller.
    assert wagers[0] > wagers[-1] > 0
    assert wagers[0] == pytest.approx(_gamma(0) * proc.initial_wealth)


def test_wealth_floors_at_zero() -> None:
    proc = OnlineFDR(level=0.05)
    for _ in range(2000):
        st = proc.update(0.999)
        assert st.wealth >= 0.0
    # A dead lane cannot borrow future wealth.
    assert proc.wealth >= 0.0


def test_fails_closed() -> None:
    proc = OnlineFDR(level=0.05)
    with pytest.raises(ValueError, match="p_value"):
        proc.update(1.5)
    with pytest.raises(ValueError, match="level"):
        OnlineFDR(level=0.0)
    with pytest.raises(ValueError, match="payout"):
        OnlineFDR(level=0.05, payout=-1.0)


def test_report_schema() -> None:
    proc = OnlineFDR(level=0.1)
    for p in [0.0001, 0.5, 0.02]:
        proc.update(p)
    rep = proc.stream_report()
    assert rep["kind"] == "online_fdr.v1"
    assert rep["n_tests"] == 3
    assert "foster_stine_alpha_investing" in rep["evidence"]
