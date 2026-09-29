"""Tests for coverage_watch — the anytime-valid nominal-coverage audit."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.research.coverage_watch import (
    COVERAGE_AUDIT_SCHEMA,
    CoverageEProcess,
    audit_interval_coverage,
)

TAUS = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


class _GaussianModel:
    """Emits correct normal quantiles with a scale multiplier."""

    fleet_lagged_predict = False

    def __init__(self, scale_factor: float = 1.0) -> None:
        self.scale_factor = scale_factor
        self._scale = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._scale = float(np.std(y)) * self.scale_factor or 1.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        row = norm.ppf(TAUS, loc=0.0, scale=self._scale)
        return np.tile(row, (x.shape[0], 1))


class _GaussianFactory:
    def __init__(self, scale_factor: float) -> None:
        self.scale_factor = scale_factor

    def __call__(self) -> _GaussianModel:
        return _GaussianModel(self.scale_factor)


def _const_shard(n: int, seed: int):  # noqa: ANN202
    """Gaussian shard generator (n, seed) -> SyntheticShard."""
    from quant_fund.research.fleet_eval import SyntheticShard

    rng = np.random.default_rng(seed + 7)
    return SyntheticShard(
        "const",
        np.zeros((n, 1)),
        rng.normal(0.0, 1.0, n),
        {"data_label": "SYNTHETIC"},
    )


def test_ep_is_exact_evalue_under_h0() -> None:
    """Each LR factor must have E[f]=1 exactly under H0."""
    ep = CoverageEProcess(p0=0.1)
    for g in ep.alt_grid:
        p1 = min(0.999999, g * 0.1)
        e = 0.1 * ep._lr(1, p1, 0.1) + 0.9 * ep._lr(0, p1, 0.1)
        assert e == pytest.approx(1.0)


def test_ep_alarms_on_undercoverage() -> None:
    ep = CoverageEProcess(p0=0.1, alpha=0.05)
    rng = np.random.default_rng(0)
    for _ in range(400):
        ep.update(bool(rng.random() < 0.25))  # 25% breach vs 10% nominal
    assert ep.alarmed
    assert ep.breach_rate == pytest.approx(0.25, abs=0.05)


def test_ep_alarms_on_overcoverage() -> None:
    """A too-wide band is also a broken claim — the mixture must catch it."""
    ep = CoverageEProcess(p0=0.1, alpha=0.05)
    rng = np.random.default_rng(0)
    for _ in range(400):
        ep.update(bool(rng.random() < 0.02))  # 2% breach vs 10% nominal
    assert ep.alarmed


def test_ep_quiet_under_correct_coverage() -> None:
    misses = 0
    for seed in range(20):
        ep = CoverageEProcess(p0=0.1, alpha=0.05)
        rng = np.random.default_rng(seed)
        for _ in range(300):
            ep.update(bool(rng.random() < 0.1))
        misses += int(ep.alarmed)
    assert misses <= 2  # MC bound at alpha=0.05


def test_ep_fails_closed() -> None:
    with pytest.raises(ValueError):
        CoverageEProcess(alpha=0.0)
    with pytest.raises(ValueError):
        CoverageEProcess(p0=0.0)
    with pytest.raises(ValueError):
        CoverageEProcess(alt_grid=[])


def test_audit_flags_narrow_and_wide_bands() -> None:
    frame, receipt = audit_interval_coverage(
        {
            "calibrated": _GaussianFactory(1.0),
            "too_narrow": _GaussianFactory(0.5),
            "too_wide": _GaussianFactory(3.0),
        },
        shards={"const": _const_shard},
        levels=(0.9,),
        n_train=512,
        n_eval=320,
        seed=0,
    )
    assert receipt["schema"] == COVERAGE_AUDIT_SCHEMA
    by_head = {r["head"]: r for r in frame.iter_rows(named=True) if r["level"] == 0.9}
    assert not by_head["calibrated"]["coverage_alarm"]
    assert by_head["too_narrow"]["coverage_alarm"]
    assert by_head["too_narrow"]["breach_rate"] > 0.1
    assert by_head["too_wide"]["coverage_alarm"]
    assert by_head["too_wide"]["breach_rate"] < 0.1
    assert np.isfinite(by_head["too_narrow"]["alarm_origin"])


def test_audit_fails_closed() -> None:
    with pytest.raises(ValueError):
        audit_interval_coverage({}, shards={"const": _const_shard})
    with pytest.raises(ValueError):
        audit_interval_coverage(
            {"h": _GaussianFactory(1.0)}, shards={"const": _const_shard}, n_eval=0
        )
    with pytest.raises(ValueError):
        audit_interval_coverage({"h": _GaussianFactory(1.0)}, shards=[])


def test_receipt_shape() -> None:
    _, receipt = audit_interval_coverage(
        {"h": _GaussianFactory(1.0)}, shards={"const": _const_shard}, levels=(0.8,), n_eval=64
    )
    for key in ("schema", "kind", "level", "inputs_sha256", "params", "evidence"):
        assert key in receipt
    assert receipt["params"]["levels"] == [0.8]
    assert "bernoulli_lr_bet" in receipt["evidence"]


def test_strict_breach_flag_rejects_missing_and_nonbinary() -> None:
    """None is a missing observation, not a non-breach — folding it in would
    deflate the measured breach rate."""
    proc = CoverageEProcess(alpha=0.05, p0=0.1)
    for bad in (None, 2, "yes", float("nan"), 0.5):
        with pytest.raises(ValueError):
            proc.update(bad)
    for ok in (True, False, 0, 1, np.bool_(True), np.int64(1)):
        proc.update(ok)
