"""Tests for research/coverage_floor.py — hedged-wallet time-uniform
coverage floor (one-sided lower bound on true coverage via a wallet-mixture
e-process on the breach stream). SYNTHETIC validation only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.research.coverage_floor import (
    COVERAGE_FLOOR_SCHEMA,
    CoverageFloor,
    audit_coverage_floor,
    coverage_floor_bench,
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
    def __init__(self, scale_factor: float = 1.0) -> None:
        self.scale_factor = scale_factor

    def __call__(self) -> _GaussianModel:
        return _GaussianModel(self.scale_factor)


def _const_shard(n: int, seed: int):  # noqa: ANN202
    from quant_fund.research.fleet_eval import SyntheticShard

    rng = np.random.default_rng(seed + 7)
    return SyntheticShard(
        "const", np.zeros((n, 1)), rng.normal(0.0, 1.0, n), {"data_label": "SYNTHETIC"}
    )


def _stream(p: float, n: int, seed: int) -> list[bool]:
    rng = np.random.default_rng(seed)
    return [bool(x) for x in rng.random(n) < p]


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        CoverageFloor(alpha=0.0)
    with pytest.raises(ValueError):
        CoverageFloor(alpha=1.0)
    with pytest.raises(ValueError):
        CoverageFloor(p0_grid=())
    with pytest.raises(ValueError):
        CoverageFloor(p0_grid=(0.0, 0.5))
    with pytest.raises(ValueError):
        CoverageFloor(wallet_fracs=())
    with pytest.raises(ValueError):
        CoverageFloor(wallet_fracs=(0.0, 0.5))
    cf = CoverageFloor()
    with pytest.raises(ValueError):
        cf.update(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cf.update(2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cf.alarm(0.0)
    with pytest.raises(ValueError):
        cf.alarm(1.0)
    with pytest.raises(ValueError):
        coverage_floor_bench(n_stream=0)
    with pytest.raises(ValueError):
        coverage_floor_bench(alpha=2.0)
    with pytest.raises(ValueError):
        coverage_floor_bench(p_true=(1.5,))


def test_floor_is_lower_bound_on_coverage() -> None:
    """Null-validity MC pin: across seeds, the coverage floor must exceed
    the true coverage at most ~alpha of the time (time-uniform, one-sided)."""
    rng = np.random.default_rng(0)
    p = 0.10
    n_seeds, n_len = 200, 300
    violations = 0
    for _s in range(n_seeds):
        draws = rng.random(n_len) < p
        cf = CoverageFloor(alpha=0.05)
        for b in draws:
            f = cf.update(bool(b))
        if math.isfinite(f) and f > 1.0 - p + 1e-12:
            violations += 1
    # Binomial tolerance at alpha=0.05: violations/200 ≪ 2*alpha
    assert violations / n_seeds <= 0.10


def test_floor_tightens_and_tracks_nominal() -> None:
    """A nominal stream keeps its floor near (under) the true coverage."""
    cf = CoverageFloor(alpha=0.05)
    last = float("nan")
    for b in _stream(0.10, 400, seed=3):
        last = cf.update(b)
    assert math.isfinite(last)
    assert last <= 1.0 - 0.10 + 1e-12
    assert last > 0.5  # not vacuously low on a nominal stream
    assert cf.n_eval == 400 and 0.0 < cf.breach_rate() < 0.3


def test_alarm_fires_on_biased_stream() -> None:
    cf = CoverageFloor(alpha=0.05)
    for b in _stream(0.45, 400, seed=1):  # 55% coverage vs 90% nominal
        cf.update(b)
    assert cf.alarm(0.90) is True


def test_degenerate_streams_edge_fallbacks() -> None:
    """All-breach and all-clean streams must hit the correct grid edges —
    the fallbacks are directional: full hi-side rejection means the bound
    is *below* the grid (floor ≈ 1), not above it."""
    clean = CoverageFloor(alpha=0.05)
    for _ in range(400):
        clean.update(False)
    assert clean.floor() > 0.95
    lo, hi = clean.interval()
    assert lo > 0.95
    dirty = CoverageFloor(alpha=0.05)
    for _ in range(400):
        dirty.update(True)
    assert dirty.floor() < 0.05
    assert dirty.alarm(0.90) is True


def test_alarm_quiet_on_nominal_stream() -> None:
    cf = CoverageFloor(alpha=0.05)
    for b in _stream(0.10, 400, seed=2):
        cf.update(b)
    assert cf.alarm(0.90) is False


def test_two_sided_interval_contains_true_rate() -> None:
    rng = np.random.default_rng(5)
    inside = 0
    n_seeds = 60
    for _ in range(n_seeds):
        cf = CoverageFloor(alpha=0.10)
        for b in rng.random(200) < 0.2:
            cf.update(bool(b))
        lo, hi = cf.interval()
        if math.isfinite(lo) and math.isfinite(hi) and lo <= 0.8 <= hi:
            inside += 1
    assert inside / n_seeds >= 0.80  # generous slack vs nominal 0.90


def test_determinism() -> None:
    a = CoverageFloor()
    b = CoverageFloor()
    for flag in _stream(0.15, 150, seed=7):
        assert a.update(flag) == b.update(flag)
    assert a.n_eval == b.n_eval and a.n_breach == b.n_breach


def test_bench_receipt_schema_and_determinism() -> None:
    out = coverage_floor_bench(p_true=(0.1,), n_stream=120, n_seeds=8, seed=0)
    rec = out["receipt"]
    assert rec["schema"] == COVERAGE_FLOOR_SCHEMA
    assert rec["kind"] == "coverage_floor"
    assert rec["data_label"] == "SYNTHETIC"
    df = out["frame"]
    assert df.shape[0] == 1
    assert 0 <= df["floor_exceeded_true"][0] <= 8
    # deterministic: same seed -> byte-identical inputs digest
    out2 = coverage_floor_bench(p_true=(0.1,), n_stream=120, n_seeds=8, seed=0)
    assert rec["inputs_sha256"] == out2["receipt"]["inputs_sha256"]


def test_audit_coverage_floor_smoke_and_schema() -> None:
    df, rec = audit_coverage_floor(
        {"calibrated": _GaussianFactory(1.0)},
        shards={"const": _const_shard},
        levels=(0.8,),
        n_train=64,
        n_eval=48,
        seed=0,
    )
    assert rec["schema"] == COVERAGE_FLOOR_SCHEMA
    assert rec["data_label"] == "SYNTHETIC"
    assert df["status"].to_list() == ["ok"]
    row = df.to_dicts()[0]
    assert row["n_eval"] == 48
    assert math.isfinite(row["coverage_floor"])
    assert 0.0 <= row["coverage_floor"] <= 1.0
    assert isinstance(row["floor_below_nominal"], bool)


def test_audit_flags_miscalibrated_head() -> None:
    """A too-narrow head's floor must drop below the nominal level."""
    df, _ = audit_coverage_floor(
        {"too_narrow": _GaussianFactory(0.4)},
        shards={"const": _const_shard},
        levels=(0.9,),
        n_train=256,
        n_eval=256,
        seed=1,
    )
    row = df.to_dicts()[0]
    assert row["floor_below_nominal"] is True


def test_audit_fail_closed() -> None:
    with pytest.raises(ValueError):
        audit_coverage_floor({})
    with pytest.raises(ValueError):
        audit_coverage_floor({"x": _GaussianFactory(1.0)}, n_train=0)
    with pytest.raises(ValueError):
        audit_coverage_floor({"x": _GaussianFactory(1.0)}, levels=(0.9999,))
