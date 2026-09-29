"""calibration_eprocess: anytime-valid PIT-uniformity audit."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate as si  # type: ignore[import-untyped]

from quant_fund.research.calibration_eprocess import (
    CalibrationEProcess,
    audit_head_calibration,
    default_channels,
)
from quant_fund.research.fleet_eval import SyntheticShard


def _make_shard(n: int, seed: int, scale: float = 0.01) -> SyntheticShard:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, 4))
    y = rng.normal(0.0, scale, size=n)
    return SyntheticShard("g", x, y, {"data_label": "SYNTHETIC", "n": n})


SHARD_G = lambda n, seed: _make_shard(n, seed)  # noqa: E731


class _GaussianHead:
    """Emits exact normal quantiles at `scale_factor * sigma_hat` of train y.

    scale_factor=1 is exactly calibrated on gaussian shards; <1 overconfident
    (too narrow -> U-shaped PIT), >1 underconfident (hump-shaped PIT).
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        shift: float = 0.0,
        taus: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    ) -> None:
        self.scale_factor = scale_factor
        self.shift = shift
        self.taus = taus
        self._sigma = 1.0

    def fit(self, x, y, **kwargs):
        self._sigma = float(np.std(np.asarray(y, dtype=float))) or 1.0
        return self

    def predict(self, x):
        from scipy.stats import norm  # type: ignore[import-untyped]

        taus = np.asarray(self.taus)
        sigma = self.scale_factor * self._sigma
        row = norm.ppf(taus, loc=self.shift, scale=sigma)
        return np.tile(row, (np.asarray(x).shape[0], 1))


class _BrokenHead:
    def fit(self, x, y, **kwargs):
        raise RuntimeError("fit exploded")


def test_bet_densities_integrate_to_one() -> None:
    for name, f in default_channels().items():
        mass = si.quad(f, 0.0, 1.0, epsabs=1e-12)[0]
        assert mass == pytest.approx(1.0, abs=1e-9), name
        grid = np.linspace(0.0, 1.0, 1001)
        assert min(f(float(u)) for u in grid) >= 0.0, name


def test_uniform_stream_no_alarm_monte_carlo() -> None:
    alarms = 0
    for seed in range(25):
        rng = np.random.default_rng(seed)
        proc = CalibrationEProcess(alpha=0.05)
        for u in rng.uniform(0, 1, size=400):
            proc.update(float(u))
        alarms += int(proc.alarmed)
    assert alarms <= 3, f"{alarms}/25 alarms under the null"


def test_u_shaped_pit_alarms_overconfident() -> None:
    # too-narrow intervals -> PIT mass at 0 and 1
    rng = np.random.default_rng(0)
    pits = np.concatenate([rng.uniform(0, 0.25, 150), rng.uniform(0.75, 1.0, 150)])
    proc = CalibrationEProcess(alpha=0.05)
    for u in pits:
        proc.update(float(u))
    assert proc.alarmed
    ch = proc.channel_wealths
    assert ch["overconf"] == max(ch.values())


def test_humped_pit_alarms_underconfident() -> None:
    rng = np.random.default_rng(0)
    pits = rng.uniform(0.35, 0.65, 300)  # intervals too wide
    proc = CalibrationEProcess(alpha=0.05)
    for u in pits:
        proc.update(float(u))
    assert proc.alarmed
    ch = proc.channel_wealths
    assert ch["underconf"] == max(ch.values())


def test_shifted_pit_alarms_loc_channel() -> None:
    rng = np.random.default_rng(0)
    pits = np.clip(rng.normal(0.62, 0.2, 300), 0, 1)  # biased forecasts
    proc = CalibrationEProcess(alpha=0.05)
    for u in pits:
        proc.update(float(u))
    assert proc.alarmed
    ch = proc.channel_wealths
    assert ch["loc_hi"] == max(ch.values())


def test_nonfinite_pit_is_inconclusive_not_crash() -> None:
    proc = CalibrationEProcess()
    proc.update(float("nan"))
    proc.update(-0.2)
    proc.update(1.5)
    assert proc.n_inconclusive == 3
    assert proc.wealth == 1.0  # wealth untouched by inconclusive steps


def test_eprocess_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        CalibrationEProcess(alpha=1.5)
    with pytest.raises(ValueError):
        CalibrationEProcess(channels={})
    with pytest.raises(ValueError, match="<= 1"):
        from quant_fund.research.calibration_eprocess import _loc_density

        _loc_density(1.5)


def test_audit_emits_receipt_and_rows() -> None:
    factories = {
        "calibrated": lambda: _GaussianHead(1.0),
        "overconfident": lambda: _GaussianHead(0.35),
        "underconfident": lambda: _GaussianHead(3.0),
        "biased": lambda: _GaussianHead(1.0, shift=0.02),
        "broken": lambda: _BrokenHead(),
    }
    frame, receipt = audit_head_calibration(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=96,
        n_eval=192,
        seed=0,
    )
    assert receipt["kind"] == "calibration_audit.v1"
    assert receipt["schema"] == "calibration_audit.v1"
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert receipt["n_shards"] == 1 and receipt["n_models"] == 5
    ok = frame.filter(frame["status"] == "ok")
    assert ok.height == 4
    err = frame.filter(frame["status"] == "error")
    assert err.height == 1
    by_name = {r["model"]: r for r in ok.iter_rows(named=True)}
    # miscalibrated heads must flag; the calibrated head must not (this seed)
    assert by_name["overconfident"]["miscalibrated"]
    assert by_name["underconfident"]["miscalibrated"]
    assert by_name["biased"]["miscalibrated"]
    assert not by_name["calibrated"]["miscalibrated"]
    # dominant channel agrees with the defect shape
    assert by_name["overconfident"]["wealth_overconf"] >= max(
        by_name["overconfident"][f"wealth_{c}"] for c in ("loc_hi", "loc_lo", "underconf")
    )
    assert by_name["underconfident"]["wealth_underconf"] >= max(
        by_name["underconfident"][f"wealth_{c}"] for c in ("loc_hi", "loc_lo", "overconf")
    )


def test_audit_fails_closed_on_bad_args() -> None:
    with pytest.raises(ValueError):
        audit_head_calibration({}, shards={"iid_gaussian": SHARD_G})
    with pytest.raises(ValueError):
        audit_head_calibration(
            {"h": lambda: _GaussianHead()}, shards={"iid_gaussian": SHARD_G}, n_eval=0
        )
    with pytest.raises(ValueError):
        audit_head_calibration({"h": lambda: _GaussianHead()}, shards={})
