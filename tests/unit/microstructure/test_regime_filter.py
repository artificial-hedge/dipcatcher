"""Tests for microstructure/regime_filter.py — Hamilton filter on the tape."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.regime_filter import (
    REGIME_FILTER_SCHEMA,
    FilterConfig,
    RegimeFilter,
    detection_delays,
    filtered_session,
    regime_filter_bench,
)
from quant_fund.microstructure.zi_lob_simulator import RegimeState, santa_fe_config


def _cfg() -> FilterConfig:
    return FilterConfig(mu=0.1, intensity=(1.0, 2.4), p_buy=(0.5, 0.62), stay=(0.99, 0.95))


def test_filter_converges_to_stress_regime() -> None:
    """A stream of fast buy-biased MOs must push the posterior to state 1."""
    f = RegimeFilter(_cfg())
    rng = np.random.default_rng(0)
    t = 0.0
    post = []
    for _ in range(150):
        t += rng.exponential(1.0 / 0.24)  # state-1 arrival rate
        f.update(t, bool(rng.random() < 0.62))
        post.append(f.belief[1])
    # a single long dt can dip the posterior; assert sustained evidence
    assert float(np.mean(post[-20:])) > 0.5


def test_filter_converges_to_calm_regime() -> None:
    f = RegimeFilter(_cfg())
    rng = np.random.default_rng(1)
    t = 0.0
    post = []
    for _ in range(150):
        t += rng.exponential(1.0 / 0.1)  # state-0 arrival rate
        f.update(t, bool(rng.random() < 0.5))
        post.append(f.belief[0])
    assert float(np.mean(post[-20:])) > 0.5


def test_prediction_mixes_beliefs() -> None:
    """Stay<1 must drag a saturated belief back toward uncertainty."""
    f = RegimeFilter(_cfg())
    for i in range(40):
        f._b = np.asarray([0.0, 1.0])
        f.update(float(i + 1) * 100.0, True)  # big dt, buy
    # after long dt + buys repeatedly, stays <1 keep mass on state 0
    assert f.belief[0] > 0.0


def test_filter_fail_closed() -> None:
    with pytest.raises(ValueError):
        RegimeFilter(FilterConfig(mu=0.0, intensity=(1.0, 2.0), p_buy=(0.5, 0.6), stay=(0.9, 0.9)))
    with pytest.raises(ValueError):
        RegimeFilter(FilterConfig(mu=0.1, intensity=(1.0, 2.0), p_buy=(0.5, 1.5), stay=(0.9, 0.9)))
    with pytest.raises(ValueError):
        RegimeFilter(_cfg(), prior=(0.3, 0.3))
    f = RegimeFilter(_cfg())
    with pytest.raises(ValueError):
        f.update(-1.0, True)
    f.update(10.0, True)
    with pytest.raises(ValueError):
        f.update(5.0, True)  # non-monotone


def test_update_neutral_keeps_rate_info() -> None:
    f = RegimeFilter(_cfg())
    f.update_neutral(1.0)  # first event: no dt info, rate prior only
    f.update_neutral(2.0)  # short dt -> leans stress
    assert f.belief[1] > 0.5
    f2 = RegimeFilter(_cfg())
    f2.update_neutral(1.0)
    f2.update_neutral(60.0)  # long dt -> leans calm
    assert f2.belief[0] > 0.5


def test_detection_delays() -> None:
    truth = np.asarray([0] * 10 + [1] * 10 + [0] * 10)
    post = np.asarray([0.1] * 10 + [0.2] * 4 + [0.8] * 6 + [0.6] * 3 + [0.2] * 7)
    d = detection_delays(truth, post, transitions=[10, 20])
    assert d == [4, 3]


def test_filtered_session_alignment() -> None:
    """Truth labels must align: posteriors at stress MOs exceed calm mean."""
    cfg = santa_fe_config(seed=5)
    out = filtered_session(
        cfg,
        states=(RegimeState("calm", 1.0, 0.5), RegimeState("stress", 2.6, 0.65)),
        stay_probs=(0.99, 0.94),
        seed=5,
        horizon=1200.0,
    )
    assert out["n_mo"] > 50
    truth, post = out["true_states"], out["posteriors"]
    if np.any(truth == 1):
        assert float(post[truth == 1].mean()) > float(post[truth == 0].mean())


def test_bench_schema_and_contract() -> None:
    r = regime_filter_bench(horizon=1200.0, seed=3)
    assert r["schema"] == REGIME_FILTER_SCHEMA
    assert r["kind"] == "regime_filter"
    assert r["data_label"] == "SYNTHETIC"
    assert r["research_only"] is True
    assert 0.0 <= r["map_accuracy"] <= 1.0
    assert len(r["payload_sha256"]) == 64
    if r["n_transitions"] > 0:
        assert r["median_detection_delay_events"] < 30


def test_bench_determinism_and_fail_closed() -> None:
    a = regime_filter_bench(horizon=600.0, seed=9)
    b = regime_filter_bench(horizon=600.0, seed=9)
    assert a["payload_sha256"] == b["payload_sha256"]
    with pytest.raises(ValueError):
        regime_filter_bench(horizon=0.0)
    with pytest.raises(ValueError):
        regime_filter_bench(stay_probs=(0.99, 1.5))
