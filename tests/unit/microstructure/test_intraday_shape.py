from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.intraday_shape import (
    _u_shape_score,
    intraday_bench,
    sim_intraday,
)


def test_u_shape_planted_u() -> None:
    profile = np.asarray([10.0, 8.0, 5.0, 3.0, 3.0, 3.0, 5.0, 8.0, 10.0])
    # edge-third mean = mean([10,8,5] + [5,8,10])/2 = 7.667; mid mean = 3
    assert _u_shape_score(profile) == pytest.approx(7.667 / 3.0, rel=0.01)


def test_u_shape_flat_is_one() -> None:
    profile = np.full(12, 5.0)
    assert _u_shape_score(profile) == pytest.approx(1.0)


def test_u_shape_inverse() -> None:
    profile = np.asarray([1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0])
    assert _u_shape_score(profile) < 0.4


def test_u_shape_degenerate() -> None:
    assert _u_shape_score(np.asarray([1.0, 2.0])) is None
    assert _u_shape_score(np.asarray([0.0] * 9)) is None


def test_sim_intraday_flat() -> None:
    out = sim_intraday(horizon=2600, n_bins=13, seed=3)
    assert 0.7 < out["activity_u_shape"] < 1.3
    assert out["mean_spread_ticks"] > 0


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        intraday_bench(tmp_path)
