"""Unit tests for quant_fund.models.dark_pool."""

from __future__ import annotations

import numpy as np

from quant_fund.models.dark_pool import detect_hidden


def _series(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    px = 100 + np.cumsum(0.1 * rng.standard_normal(n))
    sz = 100 + 20 * rng.standard_normal(n)
    sz = np.abs(sz)
    return px, sz


def test_detector_is_causal() -> None:
    """Flags at time t must not move when prints AFTER t change —
    the old version z-scored against whole-series mean/std and a
    full-series quantile, so a late volume burst rewrote earlier
    flags (look-ahead)."""
    rng = np.random.default_rng(0)
    px, sz = _series(rng, 400)
    # plant a heavy, near-static volume episode mid-series so flags
    # actually fire in the base window
    px[200:260] = px[199] + 0.001 * rng.standard_normal(60)
    sz[200:260] *= 8
    f1 = detect_hidden(px, sz, window=40)
    assert f1[200:400].sum() >= 30  # episode actually flagged in the base series
    # append an extreme future: wild swings + giant volume — global
    # baselines explode, so under look-ahead the planted flags die
    px2 = np.concatenate([px, px[-1] + 40 * rng.standard_normal(80)])
    sz2 = np.concatenate([sz, np.full(80, 1e5)])
    f2 = detect_hidden(px2, sz2, window=40)
    np.testing.assert_array_equal(f1, f2[: len(f1)])


def test_detects_planted_static_volume() -> None:
    rng = np.random.default_rng(1)
    px = 100 + np.cumsum(0.3 * rng.standard_normal(600))
    sz = np.abs(80 + 30 * rng.standard_normal(600))
    # heavy, nearly-static volume in 400:500
    px[400:500] = px[399] + 0.001 * rng.standard_normal(100)
    sz[400:500] *= 8
    flags = detect_hidden(px, sz, window=40)
    assert flags[450:495].mean() > 0.5
    assert flags[:250].mean() < 0.5
