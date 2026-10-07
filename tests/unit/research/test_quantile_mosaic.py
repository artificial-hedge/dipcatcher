"""Tests for research/quantile_mosaic.py — intra-head grid consistency."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.fleet_eval import DEFAULT_TAUS
from quant_fund.research.quantile_mosaic import (
    QUANTILE_MOSAIC_SCHEMA,
    QuantileMosaic,
    mosaic_bench,
    pit_interpolated,
    quantile_consistency,
)

TAUS = (0.1, 0.5, 0.9)


def _grid(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mid = rng.normal(0.0, 1.0, n)
    lo = mid - rng.uniform(0.5, 1.5, n)
    hi = mid + rng.uniform(0.5, 1.5, n)
    return np.column_stack([lo, mid, hi])


def test_pit_interpolated_kat() -> None:
    q = [0.0, 1.0, 2.0]
    assert pit_interpolated(TAUS, q, -1.0) == 0.0
    assert pit_interpolated(TAUS, q, 3.0) == 1.0
    assert pit_interpolated(TAUS, q, 1.0) == 0.5
    assert pit_interpolated(TAUS, q, 0.5) == pytest.approx(0.3)
    assert pit_interpolated(TAUS, q, 1.5) == pytest.approx(0.7)


def test_pit_interpolated_fail_closed() -> None:
    with pytest.raises(ValueError):
        pit_interpolated(TAUS, [0.0, 1.0], 0.5)  # wrong width
    with pytest.raises(ValueError):
        pit_interpolated(TAUS, [0.0, float("nan"), 2.0], 0.5)
    with pytest.raises(ValueError):
        pit_interpolated(TAUS, [0.0, 1.0, 2.0], float("inf"))
    with pytest.raises(ValueError):
        pit_interpolated((0.5, 0.4, 0.9), [0.0, 1.0, 2.0], 0.5)


def test_mosaic_streams_pits() -> None:
    m = QuantileMosaic(TAUS)
    assert m.update([0.0, 1.0, 2.0], 0.5) == pytest.approx(0.3)
    m.update([0.0, 1.0, 2.0], 5.0)
    assert m.n_rows == 2
    assert m.pits == [pytest.approx(0.3), 1.0]
    br = m.breach_rates()
    # row1 y=0.5 > q0=0 only; row2 y=5 > all -> rates (1.0, 0.5, 0.5)
    assert br.tolist() == [1.0, 0.5, 0.5]


def test_mosaic_crossing_count() -> None:
    m = QuantileMosaic(TAUS)
    m.update([0.0, 1.0, 2.0], 1.0)
    m.update([0.0, 3.0, 2.0], 1.0)  # crossed row
    assert m.crossing_rate == pytest.approx(0.5)


def test_mosaic_fail_closed() -> None:
    m = QuantileMosaic(TAUS)
    with pytest.raises(ValueError):
        m.update([0.0, 1.0], 0.5)
    with pytest.raises(ValueError):
        m.update([0.0, float("nan"), 2.0], 0.5)
    with pytest.raises(ValueError):
        m.update([0.0, 1.0, 2.0], float("nan"))
    with pytest.raises(ValueError):
        QuantileMosaic(())
    with pytest.raises(ValueError):
        QuantileMosaic((0.9, 0.1))  # not increasing
    with pytest.raises(ValueError):
        m.pit_histogram(bins=1)


def test_pit_histogram_and_ks() -> None:
    m = QuantileMosaic(TAUS)
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = float(rng.normal(1.0, 1.0))
        m.update([0.0, 1.0, 2.0], y)
    h = m.pit_histogram(bins=10)
    assert h.sum() == pytest.approx(1.0)
    ks = m.pit_ks()
    assert 0.0 <= ks <= 1.0
    empty = QuantileMosaic(TAUS)
    with pytest.raises(ValueError):
        empty.pit_ks()


def test_quantile_consistency_strict_and_audit() -> None:
    q = _grid(50)
    out = quantile_consistency(q, TAUS, strict=True)
    assert out["n_crossed_rows"] == 0
    assert out["min_interval_width"] > 0.0
    bad = q.copy()
    bad[3, 2] = bad[3, 1] - 0.1  # crossing in row 3
    with pytest.raises(ValueError, match="crossing"):
        quantile_consistency(bad, TAUS, strict=True)
    out2 = quantile_consistency(bad, TAUS, strict=False)
    assert out2["n_crossed_rows"] == 1


def test_quantile_consistency_fail_closed() -> None:
    with pytest.raises(ValueError):
        quantile_consistency(np.zeros((4, 2)), TAUS)  # wrong width
    with pytest.raises(ValueError):
        quantile_consistency(np.full((4, 3), np.nan), TAUS)


def test_mosaic_bench_real() -> None:
    from quant_fund.models.distribution import GaussianDistribution

    head = lambda: GaussianDistribution(list(DEFAULT_TAUS))  # noqa: E731
    rec = mosaic_bench(
        head,
        shard_names=("iid_gaussian", "regime_switch"),
        taus=DEFAULT_TAUS,
        n=256,
        seed=2,
    )
    assert rec["schema"] == QUANTILE_MOSAIC_SCHEMA
    assert rec["kind"] == "quantile_mosaic"
    assert rec["data_label"] == "SYNTHETIC"
    assert rec["n_rows"] == 256
    assert 0.0 <= rec["crossing_rate"] <= 1.0
    assert 0.0 <= rec["pit_ks"] <= 1.0
    assert len(rec["pit_histogram"]) == 10
    assert len(rec["breach_rates"]) == len(DEFAULT_TAUS)
    assert len(rec["payload_sha256"]) == 64


def test_mosaic_bench_determinism() -> None:
    from quant_fund.models.distribution import GaussianDistribution

    head = lambda: GaussianDistribution(list(DEFAULT_TAUS))  # noqa: E731
    r1 = mosaic_bench(head, shard_names=("iid_gaussian",), n=128, seed=4)
    r2 = mosaic_bench(head, shard_names=("iid_gaussian",), n=128, seed=4)
    assert r1["payload_sha256"] == r2["payload_sha256"]


def test_mosaic_bench_fail_closed() -> None:
    from quant_fund.models.distribution import GaussianDistribution

    head = lambda: GaussianDistribution(list(DEFAULT_TAUS))  # noqa: E731
    with pytest.raises(ValueError):
        mosaic_bench(head, shard_names=("iid_gaussian",), n=4)
    with pytest.raises(ValueError):
        mosaic_bench(head, shard_names=("nonexistent_shard",), n=32)


def test_consistency_rejects_empty_grid() -> None:
    """SYNTHETIC: an empty quantile grid must fail loudly, not 0/0 crash."""
    from quant_fund.research.quantile_mosaic import quantile_consistency

    empty = np.zeros((0, 3), dtype=float)
    taus = [0.25, 0.5, 0.75]
    for strict in (True, False):
        with pytest.raises(ValueError, match="at least one row"):
            quantile_consistency(empty, taus, strict=strict)
