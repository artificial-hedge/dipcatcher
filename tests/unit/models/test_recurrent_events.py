"""Tests for recurrent-event counting-process models."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.recurrent_events import (
    andersen_gill,
    bench_recurrent,
    mcf,
    pwp_gap,
    wlw_marginal,
)


def _panel(n_subj: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_subj, 1))
    subj, st, sp, ev, evno = [], [], [], [], []
    for i in range(n_subj):
        rate = 0.5 * np.exp(0.5 * x[i, 0])
        t, prev, k = 0.0, 0.0, 0
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= 8.0:
                break
            subj.append(i)
            st.append(prev)
            sp.append(t)
            ev.append(1.0)
            evno.append(k)
            prev, k = t, k + 1
        subj.append(i)
        st.append(prev)
        sp.append(8.0)
        ev.append(0.0)
        evno.append(k)
    return (
        np.asarray(x[subj]),
        np.asarray(subj),
        np.asarray(st),
        np.asarray(sp),
        np.asarray(ev),
        np.asarray(evno),
    )


def test_mcf_linear_rate():
    rng = np.random.default_rng(1)
    subj, times, ends = [], [], np.full(80, 10.0)
    for i in range(80):
        t = 0.0
        while True:
            t += rng.exponential(2.0)  # rate 0.5
            if t >= 10.0:
                break
            subj.append(i)
            times.append(t)
    m = mcf(np.asarray(subj), np.asarray(times), ends)
    assert abs(m["mcf"][-1] - 5.0) < 1.0  # E[N(10)] = 5
    assert np.all(np.diff(m["mcf"]) >= -1e-12)  # monotone
    assert (m["se"] >= 0).all()


def test_andersen_gill_recovers_beta():
    x, subj, st, sp, ev, _ = _panel(80, seed=2)
    r = andersen_gill(x, subj, st, sp, ev)
    assert abs(r["beta"][0] - 0.5) < 0.3
    assert r["se"][0] > 0


def test_pwp_and_wlw_shapes():
    x, subj, st, sp, ev, evno = _panel(50, seed=3)
    pwp = pwp_gap(x, subj, st, sp, ev, evno)
    assert np.isfinite(pwp["beta"]).all()
    etype = np.where(ev > 0, (evno % 2), 0)
    w = wlw_marginal(x, subj, st, sp, ev, etype)
    assert w["betas"].shape[1] == 1


def test_fail_closed():
    x, subj, st, sp, ev, _ = _panel(10, seed=4)
    with pytest.raises(ValueError):
        mcf(np.array([99]), np.array([1.0]), np.array([5.0]))
    with pytest.raises(ValueError):
        andersen_gill(np.full((sp.size, 1), np.nan), subj, st, sp, ev)


def test_bench():
    res = bench_recurrent()
    assert res["synthetic_score"] == 1.0
    assert res["synthetic_ag_beta_err"] < 0.2
    assert res["synthetic_mcf_end"] > 2.0
