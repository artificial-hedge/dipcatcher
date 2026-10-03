"""Tests for microstructure/excursion_geometry.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.excursion_geometry import (
    EXCURSION_GEO_SCHEMA,
    excursion_geometry_bench,
    excursion_stats,
    excursions,
    first_passage,
)


def _path() -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(13, dtype=np.float64)
    q = np.asarray([0, 1, 3, 4, 2, 0, -5, -6, -2, 0, 7, 3, 0], dtype=np.float64)
    return times, q


def test_excursions_decomposition() -> None:
    times, q = _path()
    eps = excursions(q, times, level=3.0)
    assert len(eps) == 3
    assert eps[0].peak_abs == pytest.approx(4.0) and eps[0].side == 1
    assert eps[0].duration == pytest.approx(1.0)  # t=2..3
    assert eps[1].peak_abs == pytest.approx(6.0) and eps[1].side == -1
    assert eps[2].peak_abs == pytest.approx(7.0) and eps[2].side == 1


def test_first_passage() -> None:
    times, q = _path()
    assert first_passage(q, times, 5.0) == pytest.approx(6.0)
    assert math.isnan(first_passage(q, times, 100.0))


def test_excursion_stats_aggregates() -> None:
    times, q = _path()
    st = excursion_stats(q, times, 3.0)
    assert st["n_excursions"] == 3.0
    assert st["max_peak_abs"] == pytest.approx(7.0)
    assert st["frac_time_above"] == pytest.approx(np.mean(np.abs(q) >= 3))
    assert st["excursion_rate"] == pytest.approx(3.0 / 12.0)


def test_fail_closed_edges() -> None:
    times, q = _path()
    with pytest.raises(ValueError):
        excursions(q, times, 0.0)
    with pytest.raises(ValueError):
        first_passage(q, times, -1.0)
    with pytest.raises(ValueError):
        excursion_stats(q[:3], times[:2], 3.0)  # length mismatch
    q_bad = q.copy()
    q_bad[4] = np.inf
    with pytest.raises(ValueError):
        excursions(q_bad, times, 3.0)
    t_bad = times.copy()
    t_bad[3] = t_bad[2]  # non-increasing
    with pytest.raises(ValueError):
        excursion_stats(q, t_bad, 3.0)


def test_bench_regime_contrast() -> None:
    """Directional tape must lift time-above-level vs symmetric flow."""
    r = excursion_geometry_bench(horizon=800.0, inventory_cap=10, seed=7)
    assert r["schema"] == EXCURSION_GEO_SCHEMA
    assert r["kind"] == "excursion_geometry"
    assert r["data_label"] == "SYNTHETIC"
    sym = r["per_regime"]["symmetric"]
    bia = r["per_regime"]["buy_biased"]
    assert bia["mean_abs_inventory"] > sym["mean_abs_inventory"]
    assert r["frac_time_above_ratio_level1"] > 1.0
    assert len(r["payload_sha256"]) == 64


def test_bench_determinism_and_fail_closed() -> None:
    a = excursion_geometry_bench(horizon=400.0, inventory_cap=8, levels=(1, 2), seed=3)
    b = excursion_geometry_bench(horizon=400.0, inventory_cap=8, levels=(1, 2), seed=3)
    assert a["payload_sha256"] == b["payload_sha256"]
    with pytest.raises(ValueError):
        excursion_geometry_bench(horizon=0.0)
    with pytest.raises(ValueError):
        excursion_geometry_bench(levels=())
    with pytest.raises(ValueError):
        excursion_geometry_bench(levels=(99,))
