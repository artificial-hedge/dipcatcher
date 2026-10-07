"""Pins for the nested-quantile tail-depth e-process."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as st  # type: ignore[import-untyped]

from quant_fund.research.fleet_eval import SyntheticShard
from quant_fund.research.tail_watch import (
    TailDepthEProcess,
    audit_tail_depth,
)

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def test_exact_evalue_per_outer_step() -> None:
    """LR is an exact e-value: E[f]=1 under b ~ Bern(p0) on outer rows."""
    proc = TailDepthEProcess(alpha=0.05, p0=0.5, alt_grid=(1.4,))
    rng = np.random.default_rng(0)
    # simulate many single-bet expectations under the null
    draws = rng.uniform(0, 1, size=200_000)
    contributions = np.where(draws < 0.5, proc._lr(True, 0.7, 0.5), proc._lr(False, 0.7, 0.5))
    assert contributions.mean() == pytest.approx(1.0, abs=2e-3)


def test_null_no_alarm_monte_carlo() -> None:
    """Under a correct tail, alarms should be ~never at alpha=0.05."""
    alarms = 0
    for seed in range(25):
        rng = np.random.default_rng(seed)
        proc = TailDepthEProcess(alpha=0.05, p0=0.5)
        # stream: each row breaches q_hi w.p. 0.10; conditional deep share 0.5
        u = rng.uniform(0, 1, size=400)
        for ui in u:
            proc.update(ui < 0.10, ui < 0.05)
        alarms += int(proc.alarmed)
    assert alarms <= 2  # 25 * 0.05 = 1.25 expected


def test_thin_tail_alarms() -> None:
    """Breaches land deeper than reported: deep share >> p0 -> alarm."""
    rng = np.random.default_rng(0)
    proc = TailDepthEProcess(alpha=0.05, p0=0.5)
    u = rng.uniform(0, 1, size=400)
    for ui in u:
        # outer breach on u<0.10, but conditional deep share = 0.8 (thin tail)
        proc.update(ui < 0.10, ui < 0.08)
    assert proc.alarmed
    assert proc.deep_share > 0.65  # ~40 outer rows; true share 0.8 vs null 0.5


def test_fat_tail_alarms() -> None:
    """Breaches hug the shallow cell: deep share << p0 -> alarm."""
    rng = np.random.default_rng(0)
    proc = TailDepthEProcess(alpha=0.05, p0=0.5)
    u = rng.uniform(0, 1, size=600)
    for ui in u:
        proc.update(ui < 0.10, ui < 0.025)  # deep share 0.25 vs 0.5
    assert proc.alarmed


def test_non_outer_rows_are_neutral() -> None:
    proc = TailDepthEProcess(alpha=0.05, p0=0.5)
    for _ in range(1000):
        proc.update(False, False)
    assert proc.evalue == pytest.approx(1.0)
    assert proc._n == 1000
    assert proc._n_outer == 0


def test_fail_closed_params() -> None:
    with pytest.raises(ValueError):
        TailDepthEProcess(alpha=1.5)
    with pytest.raises(ValueError):
        TailDepthEProcess(p0=1.2)
    with pytest.raises(ValueError):
        TailDepthEProcess(alt_grid=())


class _GaussianFactory:
    """Head that emits the N(0, scale) quantiles on the shared tau grid."""

    def __init__(self, scale: float) -> None:
        self._scale = scale

    def __call__(self) -> object:
        scale = self._scale

        class _Model:
            fleet_lagged_predict = False

            def fit(self, x: np.ndarray, y: np.ndarray) -> None:
                pass

            def predict(self, x: np.ndarray) -> np.ndarray:
                row = st.norm.ppf(TAUS, scale=scale)
                return np.tile(row, (x.shape[0], 1))

        return _Model()


def _gauss_shard(n: int, seed: int) -> SyntheticShard:
    rng = np.random.default_rng(seed)
    return SyntheticShard(
        name="g",
        x=np.zeros((n, 1)),
        y=rng.normal(0.0, 1.0, n),
        config={"data_label": "SYNTHETIC"},
    )


def _heavy_shard(n: int, seed: int) -> SyntheticShard:
    rng = np.random.default_rng(seed)
    return SyntheticShard(
        name="heavy",
        x=np.zeros((n, 1)),
        y=rng.standard_t(df=3, size=n),
        config={"data_label": "SYNTHETIC"},
    )


def test_audit_end_to_end_calibrated_vs_thin() -> None:
    """A matched head should not alarm; a head reporting a thinner tail
    than realized should alarm."""
    frame_cal, receipt = audit_tail_depth(
        {"cal": _GaussianFactory(1.0)},
        shards={"g": _gauss_shard},
        n_train=64,
        n_eval=500,
        seed=0,
    )
    assert receipt["schema"] == "tail_audit.v1"
    row = frame_cal.row(0, named=True)
    assert row["status"] == "ok"
    assert not row["tail_alarm"]

    frame_thin, _ = audit_tail_depth(
        {"understated": _GaussianFactory(1.0)},
        shards={"heavy": _heavy_shard},
        n_train=64,
        n_eval=2000,
        seed=0,
    )
    row = frame_thin.row(0, named=True)
    assert row["status"] == "ok"
    assert row["tail_alarm"]
    assert row["deep_share"] > 0.5  # realized breaches land deeper


def test_audit_fail_closed_on_bad_head() -> None:
    class _BadFactory:
        def __call__(self) -> object:
            class _Model:
                fleet_lagged_predict = False

                def fit(self, x: np.ndarray, y: np.ndarray) -> None:
                    pass

                def predict(self, x: np.ndarray) -> np.ndarray:
                    return np.full((x.shape[0], len(TAUS)), np.nan)

            return _Model()

    frame, receipt = audit_tail_depth(
        {"bad": _BadFactory()},
        shards={"g": _gauss_shard},
        n_train=64,
        n_eval=200,
        seed=0,
    )
    row = frame.row(0, named=True)
    assert row["status"] == "inconclusive"  # all rows malformed -> no evidence
    assert not row["tail_alarm"]
    assert receipt["schema"] == "tail_audit.v1"


def test_strict_flags_reject_missing_and_nonbinary() -> None:
    proc = TailDepthEProcess(alpha=0.05, p0=0.5)
    for a, b in ((None, False), (True, None), (2, True), (True, "x"), (float("nan"), False)):
        with pytest.raises(ValueError):
            proc.update(a, b)
    for ok in (True, False, 0, 1, np.bool_(True), np.int64(1)):
        proc.update(ok, False)


def test_tail_receipt_v2_round_trip(tmp_path) -> None:
    """receipt_version=2 seals the tail_audit.v1 body in the envelope."""
    import json

    from quant_fund.research.receipt_v2 import verify_receipt_file
    from quant_fund.research.tail_watch import write_tail_receipt

    _, receipt = audit_tail_depth(
        {"cal": _GaussianFactory(1.0)},
        shards={"g": _gauss_shard},
        n_train=64,
        n_eval=200,
        seed=0,
    )
    path = write_tail_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == "tail_audit"
    assert payload["payload"]["inputs_sha256"] == receipt["inputs_sha256"]
    assert verify_receipt_file(path)["valid"] is True


def test_deep_breach_without_outer_fails_closed() -> None:
    """deep ⊆ outer by construction (q_lo < q_hi): a deep flag on a
    non-outer row is impossible input — likely swapped args upstream —
    and must raise rather than silently drop the event."""
    proc = TailDepthEProcess(alpha=0.05, p0=0.5)
    with pytest.raises(ValueError):
        proc.update(False, True)
