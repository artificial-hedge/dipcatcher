"""Tests for coverage_cs — the time-uniform coverage confidence sequence."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.research.coverage_cs import (
    COVERAGE_CS_SCHEMA,
    CoverageCS,
    audit_coverage_cs,
)

TAUS = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


class _GaussianModel:
    fleet_lagged_predict = False

    def __init__(self, scale_factor: float = 1.0) -> None:
        self.scale_factor = scale_factor
        self._scale = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._scale = float(np.std(y)) * self.scale_factor or 1.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.tile(norm.ppf(TAUS, 0.0, self._scale), (x.shape[0], 1))


class _GaussianFactory:
    def __init__(self, scale_factor: float) -> None:
        self.scale_factor = scale_factor

    def __call__(self) -> _GaussianModel:
        return _GaussianModel(self.scale_factor)


def _const_shard(n: int, seed: int):  # noqa: ANN202
    from quant_fund.research.fleet_eval import SyntheticShard

    rng = np.random.default_rng(seed + 7)
    return SyntheticShard(
        "const", np.zeros((n, 1)), rng.normal(0.0, 1.0, n), {"data_label": "SYNTHETIC"}
    )


def test_empty_cs_covers_all() -> None:
    cs = CoverageCS()
    lo, hi = cs.interval()
    assert lo == pytest.approx(0.01)
    assert hi == pytest.approx(0.99)


def test_cs_contains_true_rate() -> None:
    """The CS must cover the true breach rate across seeds AND time."""
    misses = 0
    for seed in range(15):
        cs = CoverageCS(alpha=0.05)
        rng = np.random.default_rng(seed)
        for _ in range(200):
            lo, hi = cs.update(bool(rng.random() < 0.10))
            if np.isfinite(lo) and not (lo <= 0.10 <= hi):
                misses += 1
                break  # time-uniform: one miss is the whole run's miss
    assert misses <= 2


def test_cs_excludes_wrong_rate() -> None:
    cs = CoverageCS(alpha=0.05)
    rng = np.random.default_rng(0)
    for _ in range(400):
        lo, hi = cs.update(bool(rng.random() < 0.30))
    assert lo <= 0.30 <= hi
    assert lo > 0.10  # nominal rate firmly excluded


def test_cs_shrinks_with_n() -> None:
    rng = np.random.default_rng(1)
    cs = CoverageCS()
    w50 = w400 = float("nan")
    for i in range(400):
        lo, hi = cs.update(bool(rng.random() < 0.10))
        if i == 49:
            w50 = hi - lo
        if i == 399:
            w400 = hi - lo
    assert w400 < w50


def test_cs_fails_closed() -> None:
    with pytest.raises(ValueError):
        CoverageCS(alpha=0.0)
    with pytest.raises(ValueError):
        CoverageCS(p0_grid=[])
    with pytest.raises(ValueError):
        CoverageCS(p0_grid=[-0.1])
    with pytest.raises(ValueError):
        CoverageCS(alt_grid=[0.0])


def test_evalue_at_gridpoint() -> None:
    cs = CoverageCS()
    for _ in range(300):
        cs.update(np.random.default_rng(2).random() < 0.5)
    e = cs.evalue_at(0.10)
    assert np.isfinite(e) and e >= 20.0  # 50% breach vs 10% null → e huge
    assert cs.evalue_at(0.1234) != cs.evalue_at(0.1234)  # off-grid → NaN


def test_audit_nominal_inside() -> None:
    frame, receipt = audit_coverage_cs(
        {
            "calibrated": _GaussianFactory(1.0),
            "too_narrow": _GaussianFactory(0.4),
        },
        shards={"const": _const_shard},
        levels=(0.9,),
        n_train=512,
        n_eval=400,
        seed=0,
    )
    assert receipt["schema"] == COVERAGE_CS_SCHEMA
    rows = {r["head"]: r for r in frame.iter_rows(named=True)}
    cal, nar = rows["calibrated"], rows["too_narrow"]
    assert cal["status"] == "ok"
    assert cal["nominal_inside"]  # 0.10 nominal inside the CS
    assert not nar["nominal_inside"]  # 0.10 excluded — band too narrow
    assert np.isfinite(nar["nominal_exit_origin"])
    assert nar["cs_low"] > 0.10


def test_audit_fails_closed() -> None:
    with pytest.raises(ValueError):
        audit_coverage_cs({}, shards={"const": _const_shard})
    with pytest.raises(ValueError):
        audit_coverage_cs({"h": _GaussianFactory(1.0)}, shards={"const": _const_shard}, n_eval=0)
