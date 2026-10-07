"""Unit tests for quant_fund.models._anom_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._anom_synth import anom_series, auc, windows


def test_series_deterministic_and_labeled() -> None:
    x1, y1 = anom_series(seed=2, n=200)
    x2, y2 = anom_series(seed=2, n=200)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert 0 < y1.sum() <= 200
    assert set(np.unique(y1)) <= {0, 1}


def test_windows_shapes_and_centers() -> None:
    x, _ = anom_series(seed=1, n=100)
    out, ctr = windows(x, w=10)
    assert out.shape == (91, 10, 3)
    assert ctr[0] == 5 and ctr[-1] == 95


def test_windows_rejects_hostile_widths() -> None:
    x, _ = anom_series(seed=1, n=100)
    with pytest.raises(ValueError, match="window"):
        windows(x, w=0)
    with pytest.raises(ValueError, match="window"):
        windows(x, w=101)
    with pytest.raises(ValueError, match="window"):
        windows(x, w=-3)


def test_auc_perfect_and_chance() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(s, y) == 1.0
    assert auc(np.array([0.8, 0.9, 0.1, 0.2]), y) == 0.0
    # ties count as 0.5
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5


def test_auc_degenerate_labels_returns_chance() -> None:
    assert auc(np.ones(5), np.zeros(5, dtype=np.int64)) == 0.5
    assert auc(np.ones(5), np.ones(5, dtype=np.int64)) == 0.5
