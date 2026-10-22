"""Dip-detect numba path: parity with the Python reference + speed floor.

SYNTHETIC timings only — not market evidence and not a live-trading claim.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pytest

from fx1.bench.dip import _HAVE_NUMBA, _detect_dip_events_python, detect_dip_events

pytestmark = pytest.mark.synthetic

_DEFAULT_HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


def _median_s(fn, *, repeats: int = 5) -> float:
    fn()
    samples: list[float] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
    finally:
        gc.enable()
    return float(np.median(samples))


def _dip_panel(n_symbols: int, n_days: int) -> tuple[list[list[float]], list[str]]:
    rng = np.random.default_rng(20260927)
    dates = [f"d{i:04d}" for i in range(n_days)]
    closes: list[list[float]] = []
    for _ in range(n_symbols):
        path = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=n_days)))
        closes.append(path.astype(float).tolist())
    return closes, dates


def test_dip_detect_matches_python_reference() -> None:
    closes, dates = _dip_panel(8, 400)
    for i, series in enumerate(closes):
        fast = detect_dip_events(series, dates, f"S{i:04d}")
        slow = _detect_dip_events_python(series, dates, f"S{i:04d}", 0.10, _DEFAULT_HORIZONS)
        assert len(fast) == len(slow)
        for a, b in zip(fast, slow, strict=True):
            assert a.asset == b.asset
            assert a.peak_date == b.peak_date
            assert a.trough_date == b.trough_date
            assert a.depth == pytest.approx(b.depth)
            assert a.recovered == b.recovered


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba required for dip floor")
def test_dip_detect_faster_than_python_reference() -> None:
    closes, dates = _dip_panel(40, 1000)

    def _fast() -> int:
        return sum(
            len(detect_dip_events(series, dates, f"S{i:04d}")) for i, series in enumerate(closes)
        )

    def _slow() -> int:
        return sum(
            len(_detect_dip_events_python(series, dates, f"S{i:04d}", 0.10, _DEFAULT_HORIZONS))
            for i, series in enumerate(closes)
        )

    speedup = _median_s(_slow) / _median_s(_fast)
    # Full API (validate + DipEvent build) still beats the interpreted loop;
    # the numba core alone is much faster, but the floor guards the shipped path.
    assert speedup >= 1.5, f"dip detect speedup {speedup:.2f}x below 1.5x floor"
