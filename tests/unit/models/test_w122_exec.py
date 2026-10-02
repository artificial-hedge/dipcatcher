"""Tests for wave-122 execution/microstructure SOTA: exec_rl,
smart_router, order_flow_imbalance, pg_mm, options_flow, dark_pool."""

from __future__ import annotations

import numpy as np

from quant_fund.models.dark_pool import TapeSim, bench_dark_pool, detect_hidden
from quant_fund.models.exec_rl import (
    MarketSim,
    TwapPolicy,
    bench_exec_rl,
    run_episode,
)
from quant_fund.models.options_flow import (
    auc_score,
    bench_options_flow,
    featurize,
    synth_tape,
)
from quant_fund.models.order_flow_imbalance import (
    BookSim,
    bench_order_flow_imbalance,
    ofi_events,
)
from quant_fund.models.pg_mm import GaussianPolicy, LobEnv, bench_pg_mm
from quant_fund.models.smart_router import (
    _VENUES,
    bench_smart_router,
    fit_ridge_logistic_target,
    route_order,
    synth_fills,
)


def test_exec_rl_twap_completes():
    sim = MarketSim(steps=6)
    mid = sim.episode(np.random.default_rng(0))
    r = run_episode(TwapPolicy(sim.steps), mid, sim)
    assert np.isfinite(r["shortfall_bps"])


def test_sor_alloc_respects_depth():
    rng = np.random.default_rng(0)
    X, yf, yt, _ = synth_fills(_VENUES, 500, rng)
    wf = fit_ridge_logistic_target(X, yf)
    wt = fit_ridge_logistic_target(X, yt)
    alloc = route_order(1.0, _VENUES, wf, wt)
    assert len(alloc) == len(_VENUES)
    for a, v in zip(alloc, _VENUES, strict=True):
        assert a <= v.depth + 1e-9


def test_ofi_sign():
    rng = np.random.default_rng(0)
    bp, ap, bd, ad, mid = BookSim(n=500).stream(rng)
    e = ofi_events(bp, ap, bd, ad)
    hit = np.mean(np.sign(e[:-1]) == np.sign(np.diff(mid)))
    assert hit > 0.6


def test_pg_mm_policy_shape():
    pol = GaussianPolicy()
    a = pol.act(np.array([1.0, 0.0, 0.5]), np.random.default_rng(0))
    assert a.shape == (2,)
    g = pol.grad_logp(np.array([1.0, 0.0, 0.5]), a)
    assert g.shape == pol.w.shape


def test_options_flow_auc():
    rng = np.random.default_rng(0)
    trades, y = synth_tape(800, rng)
    X = featurize(trades)
    assert X.shape == (800, 6)
    p = X[:, 1] / 100.0  # trivial scorer sanity
    assert 0.0 <= auc_score(y, p) <= 1.0


def test_dark_pool_detect_shape():
    rng = np.random.default_rng(0)
    px, sz, aggr, hidden = TapeSim(n=800).stream(rng)
    flag = detect_hidden(px, sz)
    assert flag.shape == hidden.shape
    assert set(np.unique(flag)) <= {0.0, 1.0}


def test_benches():
    for fn in (
        bench_exec_rl,
        bench_smart_router,
        bench_order_flow_imbalance,
        bench_pg_mm,
        bench_options_flow,
        bench_dark_pool,
    ):
        out = fn()
        assert out and all(k.startswith("synthetic_") for k in out)


def test_env_episode_keys():
    env = LobEnv(steps=5)
    out = env.episode(GaussianPolicy(), np.random.default_rng(0))
    assert {"wealth", "inv_var", "feats", "acts", "rewards"} <= set(out)
