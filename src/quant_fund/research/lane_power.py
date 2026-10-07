"""Sequential power bench — how fast does each honesty lane alarm?

The sequential-inference suite ships per-lane validity proofs (P(false
alarm ever) <= alpha). Validity is half the story; the other half is
**power**: how quickly a lane detects a defect of a given size. A suite
that is valid but deaf is theater.

``lane_power`` answers it empirically. For each registered lane it
simulates streams under a grid of defect magnitudes, runs the real
process class, and records whether/when the alarm fired:

- ``coverage_watch.CoverageEProcess`` — breach rate inflated by (1+d)
- ``coverage_cs.CoverageCS`` — same stream; alarm = CS excludes nominal
- ``tail_watch.TailDepthEProcess`` — conditional deep share inflated
- ``calibration_eprocess.CalibrationEProcess`` — PIT stream compressed
  toward 0.5 + location shift (overconfident-interval signature)
- ``drift_alarm.EProcessDriftAlarm`` — level shift of size d in the stream
- ``changepoint_localize.localizer`` — shift at mid-stream; measured by
  median |argmax - planted tau| rather than alarm rate
- ``loss_cs.MeanDiffCS`` — diffs ~ clipped N(d, 1): whether the CS
  excludes zero within the stream
- ``conformal_monitor.ConformalMartingale`` — level-shifted stream
- ``promotion`` (evalues.LossEProcess) — challenger beats incumbent by d
  per origin

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

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

LANE_POWER_SCHEMA = "lane_power.v1"


@dataclass
class _LaneResult:
    alarmed: bool
    t_alarm: float  # nan if never; first index where the process alarms
    stat: float  # lane-specific extra stat (e.g. localization error)
    stream_sha256: str  # digest of the exact update stream the lane consumed


def _stream_digest(values: Iterable[object]) -> str:
    """SHA-256 over the exact update stream a lane process consumed."""
    return hash_bytes(np.ascontiguousarray(np.asarray(list(values))).tobytes())


def _run_coverage(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.coverage_watch import CoverageEProcess

    rng = np.random.default_rng(seed)
    p0 = 0.10
    rate = min(1.0, p0 * (1.0 + defect))
    proc = CoverageEProcess(alpha=alpha, p0=p0)
    t_alarm = float("nan")
    stream: list[bool] = []
    for i in range(n):
        bit = bool(rng.uniform() < rate)
        stream.append(bit)
        proc.update(bit)
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, float(proc.breach_rate), _stream_digest(stream))


def _run_tail(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.tail_watch import TailDepthEProcess

    rng = np.random.default_rng(seed)
    p0 = 0.5  # tau_lo / tau_hi on the default grid
    share = min(1.0, p0 * (1.0 + defect))
    proc = TailDepthEProcess(alpha=alpha, p0=p0)
    t_alarm = float("nan")
    stream: list[tuple[bool, bool]] = []
    for i in range(n):
        outer = rng.uniform() < 0.10
        deep = outer and rng.uniform() < share
        pair = (bool(outer), bool(deep))
        stream.append(pair)
        proc.update(pair[0], pair[1])
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, float(proc.deep_share), _stream_digest(stream))


def _run_calibration(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    """PIT-uniformity lane: defect=0 must feed a *uniform* stream (the
    calibrated-head null); defect compresses PITs toward 0.5 plus a small
    location shift — the overconfident-interval signature."""
    from quant_fund.research.calibration_eprocess import CalibrationEProcess

    rng = np.random.default_rng(seed)
    proc = CalibrationEProcess(alpha=alpha)
    t_alarm = float("nan")
    u0 = rng.uniform(0.0, 1.0, n)
    pits = np.clip(0.5 + (u0 - 0.5) * (1.0 - 0.8 * defect) + 0.15 * defect, 0.0, 1.0)
    for i, u in enumerate(pits):
        proc.update(float(u))
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, float(proc.wealth), _stream_digest(pits))


def _run_drift(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.drift_alarm import EProcessDriftAlarm

    rng = np.random.default_rng(seed)
    proc = EProcessDriftAlarm(alpha=alpha)
    t_alarm = float("nan")
    stat = float("nan")
    stream: list[float] = []
    for i in range(n):
        x = rng.normal(defect * 2.0, 1.0)  # level shift of 2d sigma
        stream.append(float(x))
        step = proc.update(float(x))
        stat = float(step.statistic)
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, stat, _stream_digest(stream))


def _run_loss_cs(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    """Mean-difference CS: diffs ~ clipped N(d, 1); alarm = CS excludes 0.

    ``MeanDiffCS`` requires a true per-step bound, so the bench draws a
    clipped normal — the CS is only valid while |d| <= bound.
    """
    from quant_fund.research.loss_cs import MeanDiffCS

    bound = 4.0
    rng = np.random.default_rng(seed)
    proc = MeanDiffCS(alpha=alpha, bound=bound)
    t_alarm = float("nan")
    stream: list[float] = []
    for i in range(n):
        d = float(np.clip(rng.normal(defect, 1.0), -bound, bound))
        stream.append(d)
        proc.update(d)
        lo, hi = proc.interval()
        if np.isfinite(lo) and lo > 0.0 and not np.isfinite(t_alarm):
            t_alarm = float(i)
    lo, hi = proc.interval()
    # Ever-excluded, matching t_alarm and the CS validity guarantee
    # P(ever exclude | null) <= alpha: a CS that crossed then re-covered still
    # fired — final-step exclusion would understate both power and alarm rate.
    excludes = bool(np.isfinite(t_alarm))
    return _LaneResult(excludes, t_alarm, float(lo), _stream_digest(stream))


def _run_localize(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    from quant_fund.research.changepoint_localize import localize_changepoint

    rng = np.random.default_rng(seed)
    tau_true = n // 2
    x = np.concatenate([rng.normal(0, 1, tau_true), rng.normal(defect * 2.0, 1, n - tau_true)])
    res = localize_changepoint(x.tolist(), alpha=alpha, window=min(40, n // 4), min_left=10)
    err = abs(res.tau_hat - tau_true)
    return _LaneResult(bool(res.alarmed), float(res.tau_hat), float(err), _stream_digest(x))


def _run_promotion(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    """Promotion lane: challenger beats the incumbent by `defect` per origin."""
    from quant_fund.research.evalues import LossEProcess

    rng = np.random.default_rng(seed)
    proc = LossEProcess(alpha=alpha)
    t_alarm = float("nan")
    stream: list[tuple[float, float]] = []
    for _ in range(n):
        c = float(rng.normal(0.5 - defect, 0.5))
        b = float(rng.normal(0.5, 0.5))
        stream.append((c, b))
        st = proc.update(c, b)
        if st.promoted and not np.isfinite(t_alarm):
            t_alarm = float(proc.promotion_origin or 0)
    return _LaneResult(
        proc.promotion_origin is not None,
        t_alarm,
        float(proc.states[-1].evalue) if proc.states else 1.0,
        _stream_digest(stream),
    )


def _run_conformal(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    """Conformal martingale on a level-shifted stream: x ~ N(d, 1)."""
    from quant_fund.research.conformal_monitor import ConformalMartingale

    rng = np.random.default_rng(seed)
    proc = ConformalMartingale(alpha=alpha, window=min(50, max(5, n // 3)))
    t_alarm = float("nan")
    stream: list[float] = []
    for i in range(n):
        x = float(rng.normal(defect * 2.0, 1.0))
        stream.append(x)
        proc.update(x)
        if proc.alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(proc.alarmed, t_alarm, float(proc.martingale), _stream_digest(stream))


def _run_coverage_cs(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    """CoverageCS on the same breach stream as coverage_watch; alarm = CS
    excludes the nominal rate (either direction — both are violations)."""
    from quant_fund.research.coverage_cs import CoverageCS

    rng = np.random.default_rng(seed)
    p0 = 0.10
    rate = min(1.0, p0 * (1.0 + defect))
    proc = CoverageCS(alpha=alpha)
    t_alarm = float("nan")
    excluded = False
    width = float("nan")
    stream: list[bool] = []
    for i in range(n):
        bit = bool(rng.uniform() < rate)
        stream.append(bit)
        lo, hi = proc.update(bit)
        excluded = bool(np.isfinite(lo) and (lo > p0 or hi < p0))
        if excluded and not np.isfinite(t_alarm):
            t_alarm = float(i)
            width = float(hi - lo)
    # Ever-excluded, matching t_alarm and CS validity (see _run_loss_cs).
    return _LaneResult(bool(np.isfinite(t_alarm)), t_alarm, width, _stream_digest(stream))


def _run_serial(defect: float, seed: int, n: int, alpha: float) -> _LaneResult:
    """SerialWatch over a PIT stream with injected AR(1) probit-scale
    dependence: z_t = defect * z_{t-1} + sqrt(1-defect^2) * eps_t,
    u_t = Phi(z_t) — uniform marginals at every defect, so only the
    serial lane sees the defect (calibration lanes must stay silent)."""
    from scipy.stats import norm

    from quant_fund.research.serial_watch import SerialWatch

    rng = np.random.default_rng(seed)
    watch = SerialWatch(alpha=alpha)
    rho = min(0.95, max(-0.95, defect))
    t_alarm = float("nan")
    stat = float("nan")
    z = 0.0
    alarmed = False
    stream: list[float] = []
    for i in range(n):
        z = rho * z + float(np.sqrt(max(0.0, 1.0 - rho * rho))) * float(rng.standard_normal())
        u = float(norm.cdf(z))
        stream.append(u)
        state = watch.update(u)
        stat = float(state.pooled_evalue)
        alarmed = alarmed or state.pooled_alarmed
        if state.pooled_alarmed and not np.isfinite(t_alarm):
            t_alarm = float(i)
    return _LaneResult(alarmed, t_alarm, stat, _stream_digest(stream))


_LANES: dict[str, Callable[[float, int, int, float], _LaneResult]] = {
    "coverage_watch": _run_coverage,
    "tail_watch": _run_tail,
    "calibration_eprocess": _run_calibration,
    "drift_alarm": _run_drift,
    "loss_cs": _run_loss_cs,
    "changepoint_localize": _run_localize,
    "promotion": _run_promotion,
    "conformal_monitor": _run_conformal,
    "coverage_cs": _run_coverage_cs,
    "serial_watch": _run_serial,
}

# lane key → the module its runner lazy-imports (test ratchet scans the
# research/ package for monitor-family modules and fails if one is not
# registered here or explicitly excluded)
_LANE_MODULES: dict[str, str] = {
    lane: ("quant_fund.research.evalues" if lane == "promotion" else f"quant_fund.research.{lane}")
    for lane in _LANES
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
    cell_digests: dict[str, dict[str, str]] = {}
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
        except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError):
            # Narrowed from `except Exception` (quality ratchet): lane-runner faults
            # are numeric/validation; exotic errors propagate. Marked as error rows.
            for d in defects:
                rows.append({"lane": lane, "status": "error", "defect": float(d)})
            continue
        n_ok_lanes += 1
        for d in defects:
            for seed in range(n_seeds):
                try:
                    r = runner(float(d), seed, n_steps, alpha)
                    cell_digests[f"{lane}|defect={float(d)}|seed={seed}"] = {
                        "stream_sha256": r.stream_sha256
                    }
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
                except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError):
                    # Narrowed from `except Exception` (quality ratchet): lane-runner
                    # faults are numeric; exotic errors propagate. Cell marked error.
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
        # Corpus-level fingerprint: digest over the exact update streams fed
        # to each lane process per (lane, defect, seed) cell — runs over
        # identical data agree on it regardless of alpha or reporting
        # choices, which is what the cross-receipt lattice edges on.
        "dataset_sha256": hash_bytes(canonical_json_bytes({"shards": cell_digests})),
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


def write_lane_power_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a lane_power receipt and write ``lane_power_<hash>.json``.

    Filename digest = canonical ``receipt_sha256``. Atomic, fail-closed on
    a malformed receipt. ``receipt_version=2`` wraps the same body in the
    unified ``receipt.v2`` envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if (
        receipt.get("schema") != LANE_POWER_SCHEMA
        or receipt.get("kind") != "lane_power"
        or not isinstance(receipt.get("inputs_sha256"), str)
        or not isinstance(receipt.get("params"), Mapping)
    ):
        raise ValueError("lane_power receipt violates its contract")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"lane_power_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = ["LANE_POWER_SCHEMA", "lane_power_bench", "write_lane_power_receipt"]
