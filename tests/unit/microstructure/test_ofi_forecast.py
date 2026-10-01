"""Tests for microstructure/ofi_forecast.py — Cont–Kukanov–Stoikov OFI→Δmid.

Event-level OFI on the ZI-LOB simulator paired with subsequent k-event mid
changes, OLS with Newey–West SEs, calm-vs-trend MarkovRegimeFlow arms, and
the sealed ``ofi_forecast.v1`` receipt. All labeled SYNTHETIC correctness
validation — never market evidence, no live-trading claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.ofi_forecast import (
    OFI_FORECAST_SCHEMA,
    collect_ofi,
    event_ofi,
    ofi_bench,
    ofi_regression,
)
from quant_fund.microstructure.zi_lob_simulator import santa_fe_config
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes, receipt_tree
from quant_fund.utils.reproducibility import git_revision

# ---------------------------------------------------------------------------
# Per-event OFI convention (Cont–Kukanov–Stoikov 2014)
# ---------------------------------------------------------------------------


def test_event_ofi_sign_convention() -> None:
    # Bid improved: new best bid contributes its posted depth.
    bid, ask = event_ofi(0, 5, 3, 7, 1, 9, 3, 7)
    assert bid == 9.0 and ask == 0.0
    # Bid degraded (best bid removed): negative contribution of the old depth.
    bid, ask = event_ofi(1, 9, 3, 7, 0, 5, 3, 7)
    assert bid == -9.0 and ask == 0.0
    # Unchanged best bid: depth change only. Same convention on the ask.
    bid, ask = event_ofi(0, 5, 3, 7, 0, 8, 3, 4)
    assert bid == 3.0 and ask == -3.0
    # Ask improved (lower best ask) contributes ask depth; ask degraded
    # subtracts the old depth. Net ofi = ofi_bid - ofi_ask.
    bid, ask = event_ofi(0, 5, 3, 7, 0, 5, 2, 6)
    assert bid == 0.0 and ask == 6.0
    bid, ask = event_ofi(0, 5, 2, 6, 0, 5, 3, 7)
    assert bid == 0.0 and ask == -6.0
    # Empty side -> quote appearing reads as an improvement; disappearing
    # reads as the quote moving away.
    bid, ask = event_ofi(None, 0, 3, 7, 0, 4, 3, 7)
    assert bid == 4.0 and ask == 0.0
    bid, ask = event_ofi(0, 4, 3, 7, None, 0, 3, 7)
    assert bid == -4.0 and ask == 0.0


# ---------------------------------------------------------------------------
# Planted regression recovery
# ---------------------------------------------------------------------------


def test_planted_ofi_regression_recovers() -> None:
    rng = np.random.default_rng(3)
    n = 500
    ofi = rng.normal(0.0, 4.0, size=n)
    dmid = 0.0025 * ofi + rng.normal(0.0, 0.003, size=n)
    reg = ofi_regression(ofi, dmid, nw_lags=5)
    assert reg["n"] == n
    assert reg["slope"] == pytest.approx(0.0025, rel=0.05)
    assert abs(reg["intercept"]) < 0.002
    assert reg["r2"] > 0.9
    assert reg["slope_se_nw"] > 0.0
    assert reg["nw_lags"] == 5
    # Near-iid residuals: NW SE should sit near the OLS SE.
    assert reg["slope_se_nw"] == pytest.approx(reg["slope_se_ols"], rel=0.5)


# ---------------------------------------------------------------------------
# Sim stream: pairing consistency + determinism
# ---------------------------------------------------------------------------


def test_collect_ofi_pairs_consistent() -> None:
    hz, n = 15, 800
    st = collect_ofi(santa_fe_config(seed=0), hz, n_events=n)
    assert st.horizon == hz and st.n_events == n
    assert st.ofi_events.shape == (n,) and st.mid.shape == (n,)
    # Pair i is ofi_i vs mid_{i+hz} - mid_i over defined-mid endpoints only.
    expect = st.mid[hz:] - st.mid[:-hz]
    mask = np.isfinite(expect)
    np.testing.assert_array_equal(st.dmid, expect[mask])
    np.testing.assert_array_equal(st.ofi, st.ofi_events[:-hz][mask])
    assert st.n_pairs == int(mask.sum())
    assert 0 <= st.n_missing_mid <= n
    reg = ofi_regression(st.ofi, st.dmid)
    # OLS with intercept yields 0 <= R2 <= 1; slope is a measured float.
    assert 0.0 <= reg["r2"] <= 1.0
    assert np.isfinite(reg["slope"]) and np.isfinite(reg["slope_se_nw"])


def test_collect_ofi_determinism() -> None:
    kw = {"n_events": 600}
    a = collect_ofi(santa_fe_config(seed=7), 10, **kw)
    b = collect_ofi(santa_fe_config(seed=7), 10, **kw)
    np.testing.assert_array_equal(a.ofi, b.ofi)
    np.testing.assert_array_equal(a.dmid, b.dmid)
    np.testing.assert_array_equal(a.ofi_events, b.ofi_events)
    np.testing.assert_array_equal(a.mid, b.mid)
    c = collect_ofi(santa_fe_config(seed=8), 10, **kw)
    assert not np.array_equal(
        np.nan_to_num(a.ofi_events, nan=-1.0), np.nan_to_num(c.ofi_events, nan=-1.0)
    )


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        collect_ofi(santa_fe_config(seed=0), 10, n_events=10)  # horizon >= n_events
    with pytest.raises(ValueError):
        collect_ofi(santa_fe_config(seed=0), 0, n_events=50)  # horizon < 1
    with pytest.raises(ValueError):
        ofi_regression(np.array([]), np.array([]))  # empty stream
    with pytest.raises(ValueError):
        ofi_regression(np.ones(10), np.linspace(0.0, 1.0, 10))  # zero ofi variance
    with pytest.raises(ValueError):
        ofi_regression(np.linspace(0.0, 1.0, 10), np.ones(10))  # zero dmid variance
    with pytest.raises(ValueError):
        ofi_regression(np.zeros(5), np.zeros(4))  # misaligned
    with pytest.raises(ValueError):
        ofi_regression(np.linspace(0.0, 1.0, 10), np.full(10, np.nan))  # non-finite


# ---------------------------------------------------------------------------
# Bench receipt: schema, sealed payload hash, verbatim arms, measured claims
# ---------------------------------------------------------------------------


def test_bench_receipt_schema_and_claims() -> None:
    rec = ofi_bench(n_events=1200, horizon=20, seed=0)
    assert rec["schema"] == OFI_FORECAST_SCHEMA == "ofi_forecast.v1"
    assert rec["kind"] == "ofi_forecast"
    assert rec["data_label"] == "SYNTHETIC" and rec["label"] == "SYNTHETIC"
    assert rec["research_only"] is True
    assert rec["git_revision"] == git_revision()
    # Sealed hash recomputes over the receipt minus payload_sha256.
    unsigned = {k: v for k, v in rec.items() if k != "payload_sha256"}
    assert rec["payload_sha256"] == hash_bytes(canonical_json_bytes(receipt_tree(unsigned)))
    # Both arms verbatim, with measured regression blocks.
    assert set(rec["arms"]) == {"calm", "trend"}
    for arm in rec["arms"].values():
        assert arm["stay_probs"] == [0.97, 0.94]
        assert arm["n_pairs"] <= arm["n_events"] - arm["horizon"]
        reg = arm["regression"]
        assert reg["n"] == arm["n_pairs"]
        assert 0.0 <= reg["r2"] <= 1.0
        assert np.isfinite(reg["slope_se_nw"])
    # Claims are booleans derived from the measured R² values only.
    r2_calm = rec["arms"]["calm"]["regression"]["r2"]
    r2_trend = rec["arms"]["trend"]["regression"]["r2"]
    cmp_ = rec["claims"]["trend_r2_vs_calm"]
    assert rec["claims"]["ofi_r2_positive"] == (r2_calm > 0.0 and r2_trend > 0.0)
    assert cmp_["trend_higher"] == (r2_trend > r2_calm)
    assert cmp_["delta_r2"] == pytest.approx(r2_trend - r2_calm)


def test_bench_determinism() -> None:
    a = ofi_bench(n_events=800, horizon=15, seed=1)
    b = ofi_bench(n_events=800, horizon=15, seed=1)
    assert a["payload_sha256"] == b["payload_sha256"]
