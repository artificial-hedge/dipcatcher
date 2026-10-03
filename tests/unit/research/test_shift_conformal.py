"""Tests for research/shift_conformal.py — weighted split-conformal +
weighted-coverage e-process monitor. SYNTHETIC validation only.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from quant_fund.research.shift_conformal import (
    SHIFT_CONFORMAL_SCHEMA,
    WeightedCoverageEProcess,
    audit_shift_conformal,
    ess,
    make_logistic_tilt,
    shift_conformal_bench,
    uniform_weight,
    weighted_quantile,
)

TAUS = np.asarray([0.1, 0.5, 0.9])


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


def _regime_shard(n: int, seed: int):  # noqa: ANN202
    """Cal half is calm (sigma 0.5); the eval tail is volatile (sigma 4) —
    conformal widening fitted on cal cannot rescue a drifted regime."""
    from quant_fund.research.fleet_eval import SyntheticShard

    rng = np.random.default_rng(seed + 11)
    y = np.concatenate([rng.normal(0.0, 0.5, n // 2), rng.normal(0.0, 4.0, n - n // 2)])
    x = rng.uniform(-1.0, 1.0, (n, 1))
    return SyntheticShard("regime", x, y, {"data_label": "SYNTHETIC"})


# ---------------------------------------------------------------------------
# weighted_quantile / ess
# ---------------------------------------------------------------------------


def test_weighted_quantile_basics() -> None:
    s = [0.0, 1.0, 2.0, 3.0]
    # uniform weights == ordinary quantile semantics (first index >= level)
    assert weighted_quantile(s, [1, 1, 1, 1], 0.75) == 2.0
    # heavy weight on the largest score pulls the quantile down to it
    assert weighted_quantile(s, [0.01, 0.01, 0.01, 1.0], 0.5) == 3.0
    with pytest.raises(ValueError):
        weighted_quantile(s, [1, 1, 1, 1], 0.0)
    with pytest.raises(ValueError):
        weighted_quantile(s, [1, 1, 1], 0.5)
    with pytest.raises(ValueError):
        weighted_quantile(s, [1, 1, 1, -1], 0.5)
    assert math.isnan(weighted_quantile([], [], 0.5))
    assert math.isnan(weighted_quantile(s, [0, 0, 0, 0], 0.5))


def test_ess() -> None:
    assert ess([1.0, 1.0, 1.0, 1.0]) == 4.0
    assert ess([1.0, 0.0, 0.0, 0.0]) == 1.0
    assert math.isnan(ess([]))
    with pytest.raises(ValueError):
        ess([1.0, -0.5])


def test_weight_oracles() -> None:
    x = np.asarray([1.0, 2.0])
    assert uniform_weight(x) == 1.0
    w = make_logistic_tilt(1.0)
    assert 0.0 < w(x) < 2.0
    assert w(np.asarray([-10.0])) == pytest.approx(2.0, abs=1e-4)
    assert w(np.asarray([10.0])) < 1e-3
    with pytest.raises(ValueError):
        make_logistic_tilt(float("nan"))
    with pytest.raises(ValueError):
        make_logistic_tilt(1.0, col=-1)


# ---------------------------------------------------------------------------
# WeightedCoverageEProcess
# ---------------------------------------------------------------------------


def test_ep_fail_closed() -> None:
    with pytest.raises(ValueError):
        WeightedCoverageEProcess(alpha=0.0, w_max=4.0)
    with pytest.raises(ValueError):
        WeightedCoverageEProcess(alpha=1.0, w_max=4.0)
    with pytest.raises(ValueError):
        WeightedCoverageEProcess(alpha=0.1, w_max=0.0)
    with pytest.raises(ValueError):
        WeightedCoverageEProcess(alpha=0.1, w_max=4.0, lambda_fracs=())
    ep = WeightedCoverageEProcess(alpha=0.1, w_max=4.0)
    with pytest.raises(ValueError):
        ep.update(None, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ep.update(5.0, True)  # exceeds w_max — errors, never clips
    with pytest.raises(ValueError):
        ep.update(1.0, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ep.update(1.0, 2)  # type: ignore[arg-type]


def test_ep_null_stays_calm() -> None:
    """Nominal weighted breach rate must not trip the alarm."""
    rng = np.random.default_rng(0)
    ep = WeightedCoverageEProcess(alpha=0.1, w_max=4.0)
    ws = np.minimum(rng.lognormal(0.0, 0.5, 400), 4.0)
    bs = rng.random(400) < 0.1
    for w, b in zip(ws, bs, strict=True):
        ep.update(float(w), bool(b))
    assert not ep.alarm()
    assert ep.weighted_breach_rate() == pytest.approx(float((ws * bs).sum() / ws.sum()))


def test_ep_fires_on_weighted_shift() -> None:
    """Breaches concentrated on high-weight rows must trip the alarm."""
    rng = np.random.default_rng(1)
    ep = WeightedCoverageEProcess(alpha=0.1, w_max=4.0)
    fired = False
    for _i in range(400):
        w = float(rng.uniform(0.5, 4.0))
        # breach probability rises with weight — coverage fails where it matters
        b = bool(rng.random() < (0.05 + 0.25 * w))
        if ep.update(w, b) >= 10.0:
            fired = True
            break
    assert fired


def test_ep_determinism() -> None:
    def run() -> float:
        ep = WeightedCoverageEProcess(alpha=0.1, w_max=4.0)
        rng = np.random.default_rng(2)
        for _ in range(100):
            ep.update(float(rng.uniform(0.5, 3.0)), bool(rng.random() < 0.2))
        return ep.evalue()

    assert run() == run()


def test_ep_w_max_bound_is_contract() -> None:
    """w_max is the declared nonneg bound — exceeding it must raise."""
    ep = WeightedCoverageEProcess(alpha=0.1, w_max=2.0)
    ep.update(2.0, True)  # boundary inclusive
    with pytest.raises(ValueError):
        ep.update(2.0001, False)


# ---------------------------------------------------------------------------
# Bench + audit
# ---------------------------------------------------------------------------


def test_bench_schema_and_arms() -> None:
    out = shift_conformal_bench(n_stream=256, n_seeds=12, seed=0)
    rec = out["receipt"]
    assert rec["schema"] == SHIFT_CONFORMAL_SCHEMA
    assert rec["kind"] == "shift_conformal"
    assert rec["data_label"] == "SYNTHETIC"
    frame = out["frame"]
    control = frame.filter(pl.col("arm") == "control").row(0, named=True)
    shifted = frame.filter(pl.col("arm") == "shifted").row(0, named=True)
    assert control["alarm_rate"] <= 0.1
    assert shifted["alarm_rate"] >= 0.5


def test_audit_uniform_control() -> None:
    """Uniform weights + correct Gaussian head: no alarm expected."""
    factories = {"cal": _GaussianFactory(1.15)}
    frame, receipt = audit_shift_conformal(
        factories,
        {"const": _const_shard},
        levels=(0.8,),
        taus=TAUS,
        n_train=256,
        n_eval=256,
        seed=0,
        weight_fn=uniform_weight,
    )
    assert receipt["schema"] == SHIFT_CONFORMAL_SCHEMA
    assert receipt["kind"] == "shift_conformal_audit"
    assert receipt["data_label"] == "SYNTHETIC"
    row = frame.row(0, named=True)
    assert row["status"] == "ok"
    assert row["n_eval"] == 256
    assert row["ess"] == pytest.approx(256.0)


def test_audit_regime_shift_alarms() -> None:
    """Eval-regime variance outruns the cal-fitted widening -> alarm."""
    factories = {"drift": _GaussianFactory(1.0)}
    frame, _ = audit_shift_conformal(
        factories,
        {"regime": _regime_shard},
        levels=(0.8,),
        taus=TAUS,
        n_train=256,
        n_eval=256,
        seed=1,
        weight_fn=uniform_weight,
    )
    row = frame.row(0, named=True)
    assert row["status"] == "ok"
    assert row["alarm"]
    assert row["weighted_breach_rate"] > 0.2


def test_audit_weighted_shift_alarms() -> None:
    """Same defect class through a non-uniform weight oracle."""
    factories = {"drift": _GaussianFactory(1.0)}
    frame, _ = audit_shift_conformal(
        factories,
        {"regime": _regime_shard},
        levels=(0.8,),
        taus=TAUS,
        n_train=256,
        n_eval=256,
        seed=1,
        w_max=2.0,
        weight_fn=make_logistic_tilt(2.0),
    )
    row = frame.row(0, named=True)
    assert row["alarm"]
    assert row["ess"] < row["n_eval"]  # nonuniform weights collapse ESS


def test_audit_fail_closed() -> None:
    factories = {"a": _GaussianFactory(1.0)}
    with pytest.raises(ValueError):
        audit_shift_conformal({}, {"const": _const_shard})
    with pytest.raises(ValueError):
        audit_shift_conformal(factories, {"const": _const_shard}, n_train=32)
    with pytest.raises(ValueError):
        audit_shift_conformal(factories, {"const": _const_shard}, levels=(0.9,), taus=(0.1, 0.5))
    with pytest.raises((ValueError, TypeError)):
        audit_shift_conformal(
            factories,
            {"const": _const_shard},
            weight_fn="x",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        # weight_fn exceeding w_max surfaces as an error, not a silent clip
        audit_shift_conformal(
            factories,
            {"regime": _regime_shard},
            levels=(0.8,),
            taus=TAUS,
            n_train=128,
            n_eval=64,
            w_max=1.5,
            weight_fn=make_logistic_tilt(2.0),
        )


def test_audit_determinism() -> None:
    factories = {"a": _GaussianFactory(1.0)}
    kwargs = dict(
        levels=(0.8,), taus=TAUS, n_train=128, n_eval=64, seed=3, weight_fn=uniform_weight
    )
    f1, _ = audit_shift_conformal(factories, {"const": _const_shard}, **kwargs)
    f2, _ = audit_shift_conformal(factories, {"const": _const_shard}, **kwargs)
    assert f1.equals(f2)
