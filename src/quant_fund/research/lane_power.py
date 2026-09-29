"""Sequential power bench — how fast does each honesty lane alarm?

The sequential-inference suite ships per-lane validity proofs (P(false
alarm ever) <= alpha). Validity is half the story; the other half is
**power**: how quickly a lane detects a defect of a given size. A suite
that is valid but deaf is theater.

``lane_power`` answers it empirically. For each registered lane it
simulates streams under a grid of defect magnitudes, runs the real
process class, and records whether/when the alarm fired:

- ``coverage_watch.CoverageEProcess`` — breach rate inflated by (1+d)
- ``tail_watch.TailDepthEProcess`` — conditional deep share inflated
- ``calibration_eprocess.CalibrationEProcess`` — PIT stream biased right
- ``drift_alarm.DriftEProcess`` — level shift of size d in the stream
- ``changepoint_localize.localizer`` — shift at mid-stream; measured by
  median |argmax - planted tau| rather than alarm rate
- ``loss_cs.LossCS``-style mean-difference CS — diffs ~ N(d, 1): whether
  the CS excludes zero within the stream

Every lane is lazy-imported: on a checkout where a lane's branch hasn't
merged, that lane is reported with ``status: lane_missing`` — never
fabricated. If *no* lane resolves, the bench still emits a receipt with
``n_lanes_ok: 0`` — an honest empty measurement, not a green pass.

The ``defect = 0`` row of each lane is the false-alarm control: its
alarm rate should track alpha across seeds (and the bench surfaces it
directly in ``null_alarm_rate``).

Seals a ``lane_power.v1`` receipt. Fail closed throughout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

LANE_POWER_SCHEMA = "lane_power.v1"


@dataclass
class _LaneResult:
    alarmed: bool
    t_alarm: float  # nan if never; first index where the process alarms
    stat: float  # lane-specific extra stat (e.g. localization error)


def _run_coverage(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.coverage_watch import CoverageEProcess

    rng = np.random.default_rng(seed)
    p0 = 0.10
    rate = min(1.0, p0 * (1.0 + defect))
    proc = CoverageEProcess(alpha=alpha, p0=p0)
    t_alarm = float("nan")
    for i in range(n):
        proc.update(bool(rng.uniform() < rate))
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, float(proc.breach_rate))


def _run_tail(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.tail_watch import TailDepthEProcess

    rng = np.random.default_rng(seed)
    p0 = 0.5  # tau_lo / tau_hi on the default grid
    share = min(1.0, p0 * (1.0 + defect))
    proc = TailDepthEProcess(alpha=alpha, p0=p0)
    t_alarm = float("nan")
    for i in range(n):
        outer = rng.uniform() < 0.10
        deep = outer and rng.uniform() < share
        proc.update(bool(outer), bool(deep))
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, float(proc.deep_share))


def _run_calibration(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.calibration_eprocess import CalibrationEProcess

    rng = np.random.default_rng(seed)
    proc = CalibrationEProcess(alpha=alpha)
    t_alarm = float("nan")
    pits = np.clip(rng.normal(0.5 + defect * 0.5, 0.28, n), 0.0, 1.0)
    for i, u in enumerate(pits):
        proc.update(float(u))
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, float(proc.wealth))


def _run_drift(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.drift_alarm import DriftEProcess

    rng = np.random.default_rng(seed)
    proc = DriftEProcess(alpha=alpha)
    t_alarm = float("nan")
    for i in range(n):
        x = rng.normal(defect * 2.0, 1.0)  # level shift of 2d sigma
        proc.update(float(x))
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, float(getattr(proc, "evalue", float("nan"))))


def _run_loss_cs(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    """Mean-difference CS: diffs ~ N(d, 1); alarm = CS excludes 0."""
    try:
        from quant_fund.research.loss_cs import LossCS
    except ImportError:
        from quant_fund.research.loss_cs import LossDiffCS as LossCS  # type: ignore[no-redef]

    rng = np.random.default_rng(seed)
    proc = LossCS(alpha=alpha)  # type: ignore[call-arg]
    t_alarm = float("nan")
    for i in range(n):
        d = float(rng.normal(defect, 1.0))
        proc.update(d)
        lo, hi = proc.interval()
        if np.isfinite(lo) and lo > 0.0 and not np.isfinite(t_alarm):
            t_alarm = float(i)
    lo, hi = proc.interval()
    excludes = bool(np.isfinite(lo) and lo > 0.0)
    return _LaneResult(excludes, t_alarm, float(lo))


def _run_localize(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.changepoint_localize import localize_changepoint

    rng = np.random.default_rng(seed)
    tau_true = n // 2
    x = np.concatenate([rng.normal(0, 1, tau_true), rng.normal(defect * 2.0, 1, n - tau_true)])
    res = localize_changepoint(x, alpha=alpha, window=min(40, n // 4), min_left=10)
    err = abs(res.tau_hat - tau_true)
    return _LaneResult(bool(res.alarmed), float(res.tau_hat), float(err))


_LANES: dict[str, Callable[[float, int, int, float], _LaneResult]] = {
    "coverage_watch": _run_coverage,
    "tail_watch": _run_tail,
    "calibration_eprocess": _run_calibration,
    "drift_alarm": _run_drift,
    "loss_cs": _run_loss_cs,
    "changepoint_localize": _run_localize,
}


def lane_power_bench(
    *,
    defects: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
    n_steps: int = 400,
    n_seeds: int = 20,
    alpha: float = 0.05,
    lanes: tuple[str, ...] | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Run the power bench over lanes x defects x seeds.

    Returns (frame, receipt). Frame columns: lane, defect, seed,
    alarmed, t_alarm, stat, status (ok | lane_missing | error).
    """
    selected = lanes if lanes is not None else tuple(_LANES)
    rows: list[dict[str, object]] = []
    n_ok_lanes = 0
    for lane in selected:
        runner = _LANES.get(lane)
        if runner is None:
            rows.append({"lane": lane, "status": "unknown_lane", "defect": float("nan")})
            continue
        try:
            probe = runner(0.0, 0, min(n_steps, 64), alpha)
            _ = probe
        except ImportError:
            for d in defects:
                rows.append({"lane": lane, "status": "lane_missing", "defect": float(d)})
            continue
        except Exception:
            for d in defects:
                rows.append({"lane": lane, "status": "error", "defect": float(d)})
            continue
        n_ok_lanes += 1
        for d in defects:
            for seed in range(n_seeds):
                try:
                    r = runner(float(d), seed, n_steps, alpha)
                    rows.append(
                        {
                            "lane": lane,
                            "status": "ok",
                            "defect": float(d),
                            "seed": seed,
                            "alarmed": r.alarmed,
                            "t_alarm": r.t_alarm,
                            "stat": r.stat,
                        }
                    )
                except Exception:
                    rows.append(
                        {
                            "lane": lane,
                            "status": "error",
                            "defect": float(d),
                            "seed": seed,
                        }
                    )
    frame = pl.DataFrame(
        rows,
        schema={
            "lane": pl.String,
            "status": pl.String,
            "defect": pl.Float64,
            "seed": pl.Int64,
            "alarmed": pl.Boolean,
            "t_alarm": pl.Float64,
            "stat": pl.Float64,
        },
        strict=False,
    )
    ok = frame.filter(pl.col("status") == "ok")
    null_rates: dict[str, float] = {}
    for lane in selected:
        sub = ok.filter((pl.col("lane") == lane) & (pl.col("defect") == 0.0))
        if sub.height:
            null_rates[lane] = float(np.asarray(sub["alarmed"].to_numpy(), dtype=bool).mean())
    receipt: dict[str, object] = {
        "schema": LANE_POWER_SCHEMA,
        "kind": "lane_power",
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "defects": list(defects),
            "n_steps": n_steps,
            "n_seeds": n_seeds,
            "alpha": alpha,
            "lanes": list(selected),
        },
        "n_lanes_ok": n_ok_lanes,
        "null_alarm_rate": null_rates,
        "evidence": [
            "sequential_power_measurement",
            "defect_injection_grid",
            "null_control_row_per_lane",
            "lazy_lane_resolution",
        ],
        "claims": [
            {
                "text": (
                    "measured alarm rate/time per lane per defect size; "
                    "defect=0 rows bound the false-alarm rate by alpha"
                ),
                "kind": "empirical_synthetic",
            }
        ],
    }
    return frame, receipt


__all__ = ["LANE_POWER_SCHEMA", "lane_power_bench"]
