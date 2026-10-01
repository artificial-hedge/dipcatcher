"""Tests for microstructure/waiting_times.py — lane B4-i companion.

Inter-event / inter-execution duration analysis on the ZI-LOB sim under iid
and calm/bursty regime arms (Goh & Barabási 2008 burstiness, exponential KS,
Weibull log-log shape). Everything is **labeled SYNTHETIC** correctness
validation — never market evidence, no live-trading claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.waiting_times import (
    MIN_DURATIONS,
    MIN_EXECUTIONS,
    burstiness,
    coefficient_of_variation,
    duration_stats,
    exponential_ks_statistic,
    inter_durations,
    run_waiting_time_arm,
    run_waiting_time_arms,
    weibull_shape_loglog,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

# Forbidden *headline* metric tokens (mirrors research.catalog registry).
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")


def _all_keys(obj: object) -> list[str]:
    """Recursively collect mapping keys (dicts only; walk list/tuple values)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_all_keys(item))
    return keys


@pytest.fixture(scope="module")
def arms_receipt() -> dict:
    """Sealed iid + regime arm bundle at a pinned seed (computed once)."""
    return run_waiting_time_arms(seed=3, horizon=1000.0)


# ---------------------------------------------------------------------------
# KAT: known-answer streams
# ---------------------------------------------------------------------------


def test_poisson_stream_is_poissonian_kat() -> None:
    """i.i.d. exponential waits: B≈0, CV≈1, Weibull k≈1, small KS."""
    rng = np.random.default_rng(7)
    d = rng.exponential(0.5, size=4000)
    assert abs(burstiness(d)) < 0.06
    assert abs(coefficient_of_variation(d) - 1.0) < 0.08
    k, scale, r2 = weibull_shape_loglog(d)
    assert 0.9 < k < 1.1
    assert abs(scale - 0.5) < 0.05
    assert r2 > 0.9
    assert exponential_ks_statistic(d) < 0.05


def test_constant_stream_is_periodic() -> None:
    """A constant-gap stream is the B = -1 periodic extreme."""
    d = np.full(64, 1.5)
    assert burstiness(d) == pytest.approx(-1.0)
    assert coefficient_of_variation(d) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="non-constant"):
        weibull_shape_loglog(d)


def test_planted_bursty_stream() -> None:
    """Clusters of short waits split by long gaps: B > 0.3, CV > 1, k < 1."""
    d = np.tile(np.concatenate([np.full(20, 0.05), [5.0]]), 100)
    stats = duration_stats(d)
    assert stats["burstiness"] > 0.3
    assert stats["coefficient_of_variation"] > 1.0
    assert stats["weibull_shape"] < 1.0
    assert stats["ks_stat_exponential"] > 0.1
    rng = np.random.default_rng(1)
    assert burstiness(d) > burstiness(rng.exponential(1.0, 4000))


def test_inter_durations_contract() -> None:
    np.testing.assert_allclose(inter_durations([0.0, 1.0, 1.5, 4.0]), [1.0, 0.5, 2.5])


# ---------------------------------------------------------------------------
# Fail-closed edges
# ---------------------------------------------------------------------------


def test_fail_closed_small_stream() -> None:
    with pytest.raises(ValueError, match="at least"):
        duration_stats(np.full(10, 0.5))
    with pytest.raises(ValueError, match="at least"):
        burstiness([1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        duration_stats(np.concatenate([np.full(60, 1.0), [0.0]]))
    with pytest.raises(ValueError, match="finite"):
        duration_stats(np.concatenate([np.full(60, 1.0), [float("nan")]]))
    with pytest.raises(ValueError, match="strictly increasing"):
        inter_durations([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="at least 2"):
        inter_durations([1.0])


def test_arm_fail_closed_below_min_execs() -> None:
    with pytest.raises(ValueError, match="execs"):
        run_waiting_time_arm(
            name="tiny",
            config=ZILobConfig(seed=0),
            flow=None,
            horizon=5.0,
        )


# ---------------------------------------------------------------------------
# Simulator arms
# ---------------------------------------------------------------------------


def test_sim_arms_smoke(arms_receipt: dict) -> None:
    assert arms_receipt["label"] == "SYNTHETIC"
    assert arms_receipt["live_pnl_claim"] is False
    assert set(arms_receipt["arms"]) == {"iid", "regime"}
    for arm in arms_receipt["arms"].values():
        assert arm["n_execs"] >= MIN_EXECUTIONS
        for stream in ("inter_event", "inter_exec"):
            s = arm[stream]
            assert s["n"] >= MIN_DURATIONS
            assert -1.0 <= s["burstiness"] <= 1.0
            assert s["coefficient_of_variation"] > 0.0
            assert 0.0 <= s["ks_stat_exponential"] <= 1.0
            assert s["weibull_shape"] > 0.0
            assert math.isfinite(s["mean"]) and s["mean"] > 0.0
            assert len(s["histogram"]["counts"]) == s["histogram"]["n_bins"]
            assert sum(s["histogram"]["counts"]) == s["n"]


def test_regime_arm_is_burstier_than_iid(arms_receipt: dict) -> None:
    s_iid = arms_receipt["arms"]["iid"]["inter_exec"]
    s_reg = arms_receipt["arms"]["regime"]["inter_exec"]
    assert s_reg["burstiness"] > s_iid["burstiness"]
    assert s_reg["coefficient_of_variation"] > s_iid["coefficient_of_variation"]
    assert s_reg["weibull_shape"] < s_iid["weibull_shape"]
    regime = arms_receipt["arms"]["regime"]["regime"]
    assert regime["n_transitions"] > 0
    assert sum(regime["state_mo_counts"]) == regime["n_mo"]


def test_receipt_seal_convention(arms_receipt: dict) -> None:
    body = {k: v for k, v in arms_receipt.items() if k != "receipt_sha256"}
    assert arms_receipt["receipt_sha256"] == hash_bytes(canonical_json_bytes(body))


def test_no_forbidden_headline_tokens(arms_receipt: dict) -> None:
    keys = _all_keys(arms_receipt)
    leaked = [k for k in keys if any(tok in k.lower() for tok in FORBIDDEN_HEADLINE_TOKENS)]
    assert not leaked
    assert "pnl" not in " ".join(keys).lower().split()
    for arm in arms_receipt["arms"].values():
        assert all(not str(k).startswith("sim_internal") for k in _all_keys(arm))
