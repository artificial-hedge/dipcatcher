"""evalues: anytime-valid head promotion — martingale, causality, validity."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.evalues import LossEProcess, promotion_report


def test_identical_streams_never_promote() -> None:
    proc = LossEProcess(alpha=0.05)
    for _ in range(200):
        st = proc.update(0.5, 0.5)
    assert st.evalue == pytest.approx(1.0)
    assert st.anytime_p == pytest.approx(1.0)
    assert proc.promotion_origin is None


def test_better_challenger_promotes() -> None:
    rng = np.random.default_rng(0)
    chall = 0.5 + rng.normal(0, 0.01, size=400)
    inc = 0.5 + rng.normal(0, 0.01, size=400) + 0.005
    rep = promotion_report(chall, inc, challenger="new", incumbent="old")
    assert rep["promoted"] is True
    assert rep["anytime_p"] < 0.05
    assert rep["promotion_origin"] is not None and rep["promotion_origin"] < 400
    assert rep["mean_loss_diff"] < 0


def test_worse_challenger_shrinks() -> None:
    proc = LossEProcess()
    for _ in range(50):
        st = proc.update(1.0, 0.5)
    assert st.evalue < 0.5
    assert st.anytime_p == pytest.approx(1.0)
    assert proc.promotion_origin is None


def test_causality_prefix_invariant() -> None:
    rng = np.random.default_rng(7)
    c = rng.normal(0.5, 0.02, size=80)
    b = rng.normal(0.5, 0.02, size=80)
    proc_a = LossEProcess()
    states_full = [proc_a.update(ci, bi) for ci, bi in zip(c, b, strict=True)]
    proc_b = LossEProcess()
    states_prefix = [proc_b.update(ci, bi) for ci, bi in zip(c[:40], b[:40], strict=True)]
    # Editing the suffix cannot change any prefix state.
    for sf, sp in zip(states_full[:40], states_prefix, strict=True):
        assert sf.evalue == pytest.approx(sp.evalue)
        assert sf.anytime_p == pytest.approx(sp.anytime_p)


def test_determinism() -> None:
    rng = np.random.default_rng(3)
    c = rng.normal(0.5, 0.05, size=60).tolist()
    b = rng.normal(0.5, 0.05, size=60).tolist()
    r1 = promotion_report(c, b)
    r2 = promotion_report(c, b)
    assert r1 == r2


def test_martingale_validity_under_symmetric_null() -> None:
    """Under a symmetric zero-mean differential the promotion rate is <= alpha + slack."""
    promoted = 0
    trials = 200
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        d = rng.normal(0.0, 0.02, size=120)
        proc = LossEProcess(alpha=0.05)
        for di in d:
            proc.update(float(di), 0.0)
        if proc.promotion_origin is not None:
            promoted += 1
    # Ville bound guarantees <=5%; e-processes are conservative in practice.
    assert promoted / trials <= 0.10


def test_fails_closed_on_bad_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        promotion_report([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="nonempty"):
        promotion_report([], [])
    with pytest.raises(ValueError, match="finite"):
        promotion_report([np.nan, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="lam"):
        LossEProcess(lam=1.5)
    with pytest.raises(ValueError, match="alpha"):
        LossEProcess(alpha=0.0)


def test_report_schema() -> None:
    rng = np.random.default_rng(11)
    rep = promotion_report(rng.normal(0.4, 0.02, 50), rng.normal(0.5, 0.02, 50), alpha=0.01)
    assert rep["kind"] == "evalue_promotion.v1"
    assert rep["alpha"] == 0.01
    assert rep["n_origins"] == 50
    assert "ville_inequality" in rep["evidence"]
    assert rep["final_evalue"] > 0
