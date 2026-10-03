"""Tests for research/mean_eprocess.py — bounded-mean wallet e-process:
time-uniform CS on the mean of a [0,1] stream + share-based dominance
audit. SYNTHETIC validation only.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from quant_fund.research.mean_eprocess import (
    MEAN_EPROCESS_SCHEMA,
    MeanEProcess,
    audit_mean_eprocess,
    mean_eprocess_bench,
    share_stream,
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


def _bern(mu: float, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < mu).astype(float).tolist()


# ---------------------------------------------------------------------------
# Contract / fail-closed
# ---------------------------------------------------------------------------


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        MeanEProcess(alpha=0.0)
    with pytest.raises(ValueError):
        MeanEProcess(alpha=1.0)
    with pytest.raises(ValueError):
        MeanEProcess(mu0_grid=())
    with pytest.raises(ValueError):
        MeanEProcess(mu0_grid=(0.0, 0.5))
    with pytest.raises(ValueError):
        MeanEProcess(wallet_fracs=())
    with pytest.raises(ValueError):
        MeanEProcess(wallet_fracs=(1.0,))
    ep = MeanEProcess()
    with pytest.raises(ValueError):
        ep.update(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ep.update(1.5)
    with pytest.raises(ValueError):
        ep.update(-0.1)
    with pytest.raises(ValueError):
        ep.update(float("nan"))
    with pytest.raises(ValueError):
        ep.alarm_below(0.0)
    with pytest.raises(ValueError):
        ep.alarm_above(1.0)


def test_grid_off_grid_evalue_nan() -> None:
    ep = MeanEProcess()
    ep.update(0.4)
    assert math.isnan(ep.evalue_le(0.375))
    assert math.isnan(ep.evalue_ge(0.375))
    assert math.isfinite(ep.evalue_le(0.4))


def test_degenerate_streams_bounds() -> None:
    ep_hi = MeanEProcess()
    for _ in range(60):
        ep_hi.update(1.0)
    lo, hi = ep_hi.interval()
    assert lo > 0.9
    ep_lo = MeanEProcess()
    for _ in range(60):
        ep_lo.update(0.0)
    lo2, hi2 = ep_lo.interval()
    assert hi2 < 0.1
    assert ep_lo.alarm_below(0.5)


def test_validity_monte_carlo() -> None:
    """One-sided lower bound must cover the true mean except <= alpha."""
    mu = 0.5
    violations = 0
    n_seeds = 200
    for s in range(n_seeds):
        ep = MeanEProcess()
        for x in _bern(mu, 300, s):
            ep.update(x)
        if ep.lower_bound() > mu + 1e-12:
            violations += 1
    assert violations / n_seeds <= 0.10


def test_upper_validity_monte_carlo() -> None:
    mu = 0.3
    violations = 0
    n_seeds = 200
    for s in range(n_seeds):
        ep = MeanEProcess()
        for x in _bern(mu, 300, 10_000 + s):
            ep.update(x)
        if ep.upper_bound() < mu - 1e-12:
            violations += 1
    assert violations / n_seeds <= 0.10


def test_cs_width_shrinks() -> None:
    xs = _bern(0.5, 800, 0)
    ep = MeanEProcess()
    w_early = w_late = 0.0
    for i, x in enumerate(xs):
        lo, hi = ep.update(x)
        if i == 199:
            w_early = hi - lo
        if i == 799:
            w_late = hi - lo
    assert w_late < w_early


def test_alarm_on_shifted_mean() -> None:
    ep = MeanEProcess()
    xs = _bern(0.6, 50, 1) + _bern(0.3, 400, 2)
    fired = False
    for x in xs:
        ep.update(x)
        if ep.alarm_below(0.5):
            fired = True
    assert fired


def test_determinism() -> None:
    def run() -> tuple[float, float]:
        ep = MeanEProcess()
        for x in _bern(0.4, 200, 3):
            ep.update(x)
        return ep.interval()

    assert run() == run()


# ---------------------------------------------------------------------------
# share_stream
# ---------------------------------------------------------------------------


def test_share_stream_contract() -> None:
    xs = share_stream([1.0, 2.0, 0.0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(xs, [0.5, 2.0 / 3.0, 0.5])
    with pytest.raises(ValueError):
        share_stream([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        share_stream([float("nan")], [1.0])
    with pytest.raises(ValueError):
        share_stream([-1.0], [1.0])


# ---------------------------------------------------------------------------
# Bench + audit
# ---------------------------------------------------------------------------


def test_bench_schema_and_validity() -> None:
    out = mean_eprocess_bench(n_stream=256, n_seeds=16, seed=0)
    rec = out["receipt"]
    assert rec["schema"] == MEAN_EPROCESS_SCHEMA
    assert rec["kind"] == "mean_eprocess"
    assert rec["data_label"] == "SYNTHETIC"
    frame = out["frame"]
    assert frame.shape[0] == 3
    assert frame["exclusion_rate"].max() <= 0.2


def test_audit_dominance_verdict() -> None:
    """A head with half the noise scale should dominate on share < 0.5."""
    factories = {"wide": _GaussianFactory(2.0), "tight": _GaussianFactory(0.4)}
    frame, receipt = audit_mean_eprocess(
        factories,
        {"const": _const_shard},
        baseline="wide",
        taus=(0.1, 0.5, 0.9),
        n_train=256,
        n_eval=512,
        seed=0,
    )
    assert receipt["schema"] == MEAN_EPROCESS_SCHEMA
    assert receipt["kind"] == "mean_eprocess_audit"
    assert receipt["data_label"] == "SYNTHETIC"
    tight_rows = frame.filter((pl.col("head") == "tight") & (pl.col("tau") == 0.1))
    assert tight_rows.shape[0] == 1
    row = tight_rows.row(0, named=True)
    assert row["status"] == "ok"
    assert row["share_mean"] < 0.5
    assert row["verdict"] == "dominates"
    # at the median both heads predict identically -> share = 0.5, no verdict
    med = frame.filter((pl.col("head") == "tight") & (pl.col("tau") == 0.5)).row(0, named=True)
    assert med["share_mean"] == 0.5 and med["verdict"] == "inconclusive"


def test_audit_fail_closed() -> None:
    factories = {"a": _GaussianFactory(1.0), "b": _GaussianFactory(1.0)}
    with pytest.raises(ValueError):
        audit_mean_eprocess({}, {"const": _const_shard})
    with pytest.raises(ValueError):
        audit_mean_eprocess({"only": _GaussianFactory(1.0)}, {"const": _const_shard})
    with pytest.raises(ValueError):
        audit_mean_eprocess(
            factories,
            {"const": _const_shard},
            baseline="missing",
            n_train=64,
            n_eval=64,
        )
    with pytest.raises(ValueError):
        audit_mean_eprocess(factories, {"const": _const_shard}, taus=(1.5,), n_train=64, n_eval=64)


def test_audit_determinism() -> None:
    factories = {"a": _GaussianFactory(1.0), "b": _GaussianFactory(0.9)}
    kwargs = dict(baseline="a", taus=(0.5,), n_train=128, n_eval=128, seed=5)
    f1, _ = audit_mean_eprocess(factories, {"const": _const_shard}, **kwargs)
    f2, _ = audit_mean_eprocess(factories, {"const": _const_shard}, **kwargs)
    assert f1.equals(f2)


def test_share_stream_in_audit_is_bounded() -> None:
    """Every streamed share must lie in [0,1] — the e-factor nonneg bound."""
    lh = np.abs(np.random.default_rng(0).normal(0, 1, 50))
    lb = np.abs(np.random.default_rng(1).normal(0, 2, 50))
    xs = share_stream(lh, lb)
    assert xs.min() >= 0.0 and xs.max() <= 1.0
    ep = MeanEProcess()
    for x in xs:
        ep.update(float(x))
    assert ep.n == 50
