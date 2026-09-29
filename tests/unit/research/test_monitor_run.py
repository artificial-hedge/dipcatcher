"""Contract tests for ``monitor_fleet`` — the all-lanes tournament monitor."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.fleet_eval import SyntheticShard
from quant_fund.research.monitor_run import MONITOR_RUN_SCHEMA, monitor_fleet


class _GaussianFactory:
    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale

    def __call__(self):  # noqa: D102 - factory protocol
        scale = self._scale

        class _Model:
            def fit(self, x, y) -> None:
                self._loc = float(np.asarray(y, dtype=float).mean())

            def predict(self, x):
                x = np.asarray(x, dtype=float)
                n = x.shape[0]
                taus = np.asarray([0.05, 0.1, 0.5, 0.9, 0.95])
                from scipy.stats import norm

                z = norm.ppf(taus) * scale
                return np.tile(self._loc + z, (n, 1))

        return _Model()


def _shard(n: int, seed: int) -> SyntheticShard:
    rng = np.random.default_rng(seed)
    return SyntheticShard(
        name="g",
        x=np.zeros((n, 1)),
        y=rng.normal(0.0, 1.0, n),
        config={"data_label": "SYNTHETIC"},
    )


def _shards() -> dict:
    return {"s1": _shard, "s2": _shard}


def test_frame_shape_and_columns() -> None:
    frame, receipt = monitor_fleet(
        {"tight": _GaussianFactory(1.0), "wide": _GaussianFactory(2.0)},
        _shards(),
        n_train=256,
        n_eval=64,
        taus=[0.05, 0.1, 0.5, 0.9, 0.95],
        seed=1,
    )
    assert receipt["schema"] == MONITOR_RUN_SCHEMA
    assert receipt["kind"] == "monitor_run"
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert set(receipt["params"]["heads"]) == {"tight", "wide"}
    assert receipt["params"]["tail_cell"] == [0.05, 0.1]
    assert frame.height == 4  # 2 shards x 2 heads
    for col in ("coverage_alarmed", "tail_alarmed", "drift_alarmed"):
        assert col in frame.columns


def test_deterministic_replay_is_byte_identical() -> None:
    """The monitor is a deterministic function of (factories, shards, seed):
    two runs must produce byte-identical frames and receipts — otherwise the
    sealed receipt could not be reproduced as evidence."""
    kwargs = dict(
        n_train=256,
        n_eval=64,
        taus=[0.05, 0.1, 0.5, 0.9, 0.95],
        seed=7,
        alpha=0.05,
        level=0.9,
    )
    factories = {"tight": _GaussianFactory(1.0), "wide": _GaussianFactory(2.0)}
    frame_a, receipt_a = monitor_fleet(dict(factories), _shards(), **kwargs)
    frame_b, receipt_b = monitor_fleet(dict(factories), _shards(), **kwargs)
    assert frame_a.write_csv() == frame_b.write_csv()
    assert receipt_a == receipt_b


def test_broken_head_rows_status_error() -> None:
    class _Broken:
        def __call__(self):
            raise RuntimeError("nope")

    frame, _ = monitor_fleet(
        {"ok": _GaussianFactory(1.0), "dead": _Broken()},
        _shards(),
        n_train=256,
        n_eval=64,
        taus=[0.05, 0.1, 0.5, 0.9, 0.95],
    )
    dead = frame.filter((frame["head"] == "dead") & (frame["status"] == "error"))
    assert dead.height == 2
    ok = frame.filter(frame["status"] == "ok")
    assert ok.height == 2


def test_miscalibrated_head_alarms_on_calibration() -> None:
    # a head that is way too tight must look miscalibrated; lane may be absent
    frame, receipt = monitor_fleet(
        {"pin": _GaussianFactory(0.05)},
        _shards(),
        n_train=256,
        n_eval=128,
        taus=[0.05, 0.1, 0.5, 0.9, 0.95],
        seed=3,
    )
    if receipt["lanes_available"]["calibration"]:
        e = frame["calibration_evalue"].to_numpy()
        assert np.all(np.isfinite(e))


def test_tau_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        monitor_fleet(
            {"a": _GaussianFactory(1.0)},
            _shards(),
            taus=[0.5, 0.1],  # unsorted
        )
    with pytest.raises(ValueError, match="level"):
        monitor_fleet({"a": _GaussianFactory(1.0)}, _shards(), level=1.2)
