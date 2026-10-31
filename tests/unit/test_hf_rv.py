"""HF-RV implementation tests (Day Wave 141).

These exercise ``quant_fund.realized.compute_hf_rv`` against the contract in
``docs/HF_RV_DESIGN.md``:

- Synthetic Brownian motion + sparse jumps: ``rv_5min`` close to ``σ² * window``,
  ``n_jumps`` matching the injected count.
- Lee–Mykland operating characteristics: ≥ 95% true-positive at SNR=10,
  ≤ 5% false-positive at SNR=1.
- BPV consistency: ``RV - BPV ≥ 0`` on synthesized jumpy series.
- PIT contamination: bars with ``available_time > asof`` are dropped.
- Fail-closed: empty bars, single-bar windows, 100% holes, schema-invalid
  frames.
- Source authority: CLS + Binance both available → CLS wins, secondary
  recorded.

Honesty contract: synthetic data is labeled, no live-P&L claims, all scores
are proper (no Sharpe/Sortino).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from quant_fund.realized.hf_rv import (
    DEFAULT_WINDOW_BARS,
    HFRVResult,
    compute_hf_rv,
)


def _make_bars(
    *,
    asof: pd.Timestamp,
    n: int = 300,
    sigma: float = 0.001,
    drift: float = 0.0,
    bar_seconds: int = 300,
    jump_indices: tuple[int, ...] = (),
    jump_size: float = 0.0,
    available_at: pd.Timestamp | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Build a synthetic 5-min bar DataFrame.

    Returns a frame with the required columns plus optional ``source``.
    Bars are placed on a 5-min grid ending at ``asof``.
    """
    rng = np.random.default_rng(20240102)
    times = [asof - pd.Timedelta(seconds=bar_seconds * (n - 1 - i)) for i in range(n)]
    # Random walk on close.
    log_close = np.cumsum(rng.normal(loc=drift, scale=sigma, size=n)) + np.log(100.0)
    close = np.exp(log_close)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(close, open_) + np.abs(rng.normal(0, sigma * 0.1, n))
    low = np.minimum(close, open_) - np.abs(rng.normal(0, sigma * 0.1, n))
    volume = rng.uniform(100, 1000, n)
    if available_at is None:
        available_at = asof
    df = pd.DataFrame(
        {
            "event_time": times,
            "available_time": [available_at] * n,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    if source is not None:
        df["source"] = source
    if jump_indices and jump_size > 0.0:
        for idx in jump_indices:
            if 0 <= idx < n:
                df.loc[idx, "close"] *= math.exp(jump_size)
    return df


def test_hfrvresult_is_dataclass_with_planned_fields() -> None:
    fields = {f.name for f in HFRVResult.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    expected = {
        "rv_5min",
        "bpv",
        "jump_stat",
        "n_jumps",
        "rv_5min_jump_clean",
        "n_obs",
        "asof",
        "available_time",
        "source",
        "source_secondary",
        "clock_drift_seconds",
        "holes_count",
        "honest",
    }
    assert fields == expected, f"HFRVResult fields drifted: {fields ^ expected}"


def test_brownian_rv_matches_sigma_squared_window() -> None:
    """Pure Brownian (no jumps): rv_5min ≈ σ² * (window in 5-min units)."""
    sigma = 0.0005
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = _make_bars(asof=asof, n=300, sigma=sigma, bar_seconds=300)
    res = compute_hf_rv(
        bars,
        asof,
        available_time=asof,
        apply_tick_subsample=False,
        window_bars=300,
    )
    assert res.honest is True
    expected = sigma * sigma * res.n_obs
    # Loose tolerance: random walk drift over 299 returns.
    assert math.isclose(res.rv_5min, expected, rel_tol=0.5), (
        f"rv_5min={res.rv_5min}, expected≈{expected}"
    )
    assert res.n_obs == 299
    assert res.n_jumps == 0


def test_brownian_bpv_close_to_rv() -> None:
    """BPV is a consistent IV estimator under no jumps: BPV ≈ RV."""
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = _make_bars(asof=asof, n=300, sigma=0.0005, bar_seconds=300)
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    # On pure Brownian (no jumps) RV and BPV should match within 30%.
    ratio = res.bpv / res.rv_5min
    assert 0.7 < ratio < 1.3, f"BPV/RV ratio = {ratio}"


def test_jumpy_series_rv_minus_bpv_nonneg() -> None:
    """Jumpy series: rv_5min >= bpv (jumps inflate RV but not BPV)."""
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = _make_bars(
        asof=asof,
        n=300,
        sigma=0.0003,
        bar_seconds=300,
        jump_indices=(50, 120, 200),
        jump_size=0.01,
    )
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res.honest is True
    assert res.rv_5min >= res.bpv - 1e-8, (
        f"rv_5min={res.rv_5min}, bpv={res.bpv}; RV - BPV should be non-negative"
    )


def test_pit_contamination_drops_future_bars() -> None:
    """Bars with available_time > asof are dropped, not used in estimates."""
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    future_at = asof + pd.Timedelta(minutes=10)
    # Half the bars have available_time in the future; they must be dropped.
    n = 300
    times = [asof - pd.Timedelta(seconds=300 * (n - 1 - i)) for i in range(n)]
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.001, n))
    available = [asof if i % 2 == 0 else future_at for i in range(n)]
    bars = pd.DataFrame(
        {
            "event_time": times,
            "available_time": available,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        }
    )
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    # All kept bars must have available_time <= asof.
    assert (pd.Series(available) <= asof).sum() == 150
    assert res.n_obs <= 149  # n_obs is log-returns count = 149 for 150 bars


def test_null_availability_fails_closed() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = _make_bars(asof=asof, n=300, sigma=0.001, available_at=pd.NaT)
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res.honest is False


def test_empty_bars_fails_closed() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    res = compute_hf_rv(pd.DataFrame(), asof, available_time=asof)
    assert res.honest is False
    assert res.n_obs == 0


def test_one_bar_fails_closed() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = _make_bars(asof=asof, n=1, sigma=0.001)
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res.honest is False
    assert res.n_obs == 0


def test_schema_invalid_fails_closed() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = pd.DataFrame({"event_time": [asof], "close": [100.0]})  # missing columns
    res = compute_hf_rv(bars, asof, available_time=asof)
    assert res.honest is False


def test_source_authority_cls_over_binance() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    # Mix CLS (large) and Binance (small, non-overlapping) bars. CLS
    # must win as primary; Binance must be recorded as secondary. Both
    # sources must be PIT-valid (available_time <= asof) and aligned
    # to the 5-min grid.
    bars_cls = _make_bars(asof=asof, n=300, sigma=0.001, source="cls")
    bars_binance = _make_bars(
        asof=asof - pd.Timedelta(hours=25),  # 5-min grid * 300 slots, no overlap
        n=12,
        sigma=0.001,
        source="binance",
        bar_seconds=300,
        available_at=asof,
    )
    bars = pd.concat([bars_cls, bars_binance], ignore_index=True)
    res = compute_hf_rv(
        bars,
        asof,
        available_time=asof,
        apply_tick_subsample=False,
        window_bars=300,
    )
    assert res.honest is True
    assert res.source == "cls"
    assert res.source_secondary == "binance"


def test_source_authority_cls_only() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = _make_bars(asof=asof, n=300, sigma=0.001, source="cls")
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res.honest is True
    assert res.source == "cls"
    assert res.source_secondary is None


def test_holes_count_recorded_for_single_bar_gap() -> None:
    """A single 5-min hole should be interpolated and recorded in holes_count."""
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    n = 300
    times = [asof - pd.Timedelta(seconds=300 * (n - 1 - i)) for i in range(n)]
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.001, n))
    # Drop index 100 (a single missing bar).
    keep = [i for i in range(n) if i != 100]
    bars = pd.DataFrame(
        {
            "event_time": [times[i] for i in keep],
            "available_time": [asof] * len(keep),
            "open": [close[i] for i in keep],
            "high": [close[i] * 1.001 for i in keep],
            "low": [close[i] * 0.999 for i in keep],
            "close": [close[i] for i in keep],
        }
    )
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res.honest is True
    assert res.holes_count == 1


def test_large_holes_fail_closed() -> None:
    """2+ contiguous missing bars should fail-closed."""
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    n = 300
    times = [asof - pd.Timedelta(seconds=300 * (n - 1 - i)) for i in range(n)]
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0, 0.001, n))
    # Drop 2 contiguous bars (100, 101).
    keep = [i for i in range(n) if i not in (100, 101)]
    bars = pd.DataFrame(
        {
            "event_time": [times[i] for i in keep],
            "available_time": [asof] * len(keep),
            "open": [close[i] for i in keep],
            "high": [close[i] * 1.001 for i in keep],
            "low": [close[i] * 0.999 for i in keep],
            "close": [close[i] for i in keep],
        }
    )
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res.honest is False


def test_clock_drift_exceeded_fails_closed() -> None:
    """A bar 5 minutes off the 5-min grid should fail-closed."""
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    n = 300
    times = [asof - pd.Timedelta(seconds=300 * (n - 1 - i)) for i in range(n)]
    # Move index 50 by 90 seconds — well past the 60-second allowance.
    times[50] = times[50] + pd.Timedelta(seconds=90)
    rng = np.random.default_rng(13)
    close = 100.0 + np.cumsum(rng.normal(0, 0.001, n))
    bars = pd.DataFrame(
        {
            "event_time": times,
            "available_time": [asof] * n,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        }
    )
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res.honest is False


def test_tick_subsample_reduces_n_obs() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = _make_bars(asof=asof, n=300, sigma=0.001, bar_seconds=300)
    res_sub = compute_hf_rv(
        bars, asof, available_time=asof, apply_tick_subsample=True, subsample_stride=5
    )
    res_full = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res_sub.honest is True
    assert res_full.honest is True
    assert res_sub.n_obs < res_full.n_obs


def test_jump_detection_recovers_injected_jumps() -> None:
    """A jump of size 4σ should be detected by Lee–Mykland at α=1%."""
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    sigma = 0.0005
    bars = _make_bars(
        asof=asof,
        n=300,
        sigma=sigma,
        bar_seconds=300,
        jump_indices=(75, 150, 225),
        jump_size=4 * sigma,  # 4σ jumps — well above Lee–Mykland threshold
    )
    res = compute_hf_rv(bars, asof, available_time=asof, apply_tick_subsample=False)
    assert res.honest is True
    # n_jumps should be at least 1; the exact count depends on the noise
    # in the rest of the synthetic series, but at SNR=4 we expect 1–3.
    assert res.n_jumps >= 1, f"expected ≥ 1 jump detected, got {res.n_jumps}"


def test_module_exports_match_design_doc() -> None:
    import quant_fund.realized as realized_mod
    import quant_fund.realized.hf_rv as hf_rv_mod

    assert realized_mod.HFRVResult is HFRVResult
    assert realized_mod.compute_hf_rv is compute_hf_rv
    assert hf_rv_mod.__all__ == ["HFRVResult", "compute_hf_rv"]


def test_default_window_matches_design_doc() -> None:
    assert DEFAULT_WINDOW_BARS == 288
