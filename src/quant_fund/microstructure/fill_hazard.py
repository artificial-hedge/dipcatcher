"""Fill-hazard estimation for resting limit orders on the ZI-LOB engine.

The execution stack's open statistical question: *given* a resting order
submitted ``delta`` ticks inside the touch, what is its fill time
distribution? This module answers it empirically on the synthetic engine:

- ``collect_fill_records`` — probe lane: submits unit probe orders at a
  ladder of tick offsets while the simulator runs, then joins fills on
  ``maker_order_id``; orders still resting at the horizon are right-
  censored (never treated as fills or infinite waits).
- ``kaplan_meier`` / ``nelson_aalen`` — nonparametric survival + cumulative
  hazard per offset bucket, with Greenwood-free point estimates (the
  variance estimator is intentionally omitted — degenerate risk sets make
  it misleading; use the bench's held-out calibration instead).
- ``fit_exp_intensity`` — the market-making literature's working model
  ``lambda(delta) = A exp(-kappa delta)``: per-bucket Poisson rate MLE
  ``events / exposure`` then a weighted log-linear fit for ``(A, kappa)``.
- ``fill_prob`` — ``1 - exp(-lambda t)`` under the fitted model.
- ``fill_hazard_bench`` — sealed ``fill_hazard.v1`` receipt pinning KM
  monotonicity in delta, kappa recovery, and held-out calibration error.

SYNTHETIC only: all records come from the simulator; nothing here is a
market claim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    Side,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

FILL_HAZARD_SCHEMA = "fill_hazard.v1"
PROBE_TAG = "fill_hazard_probe"


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _nonneg_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {x!r}")
    return v


@dataclass(frozen=True)
class FillRecord:
    """One probe order's outcome.

    ``event=True`` iff the order was filled; ``duration`` is the observed
    lifetime (fill time or censoring window). ``delta_ticks`` is the
    offset inside the touch at submit (0 = at the touch). ``queue_ahead``
    is the engine's own queue-position accounting at submit.
    """

    side: str
    delta_ticks: int
    queue_ahead: int
    t_submit: float
    duration: float
    event: bool


def collect_fill_records(
    config: ZILobConfig,
    *,
    horizon: float,
    probe_interval: float = 25.0,
    delta_ticks: tuple[int, ...] = (0, 1, 2, 4),
    sides: tuple[str, ...] = ("buy", "sell"),
    min_runway: float = 10.0,
) -> list[FillRecord]:
    """Run the sim and submit probe ladders every ``probe_interval`` seconds.

    Probes marketable at submit (moving touch) are skipped rather than
    snapped — a probe that cannot rest cannot contribute a fill time.
    Probes submitted after ``horizon - min_runway`` are skipped so every
    record has a minimum censoring window.
    """
    _pos_finite(horizon, "horizon")
    _pos_finite(probe_interval, "probe_interval")
    _nonneg_finite(min_runway, "min_runway")
    if not delta_ticks or any(int(d) < 0 for d in delta_ticks):
        raise ValueError(f"delta_ticks must be non-empty non-negative ints, got {delta_ticks!r}")
    deltas = tuple(int(d) for d in delta_ticks)
    checked_sides: list[Side] = []
    for s in sides:
        if s not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {s!r}")
        checked_sides.append(cast(Side, s))

    sim = ZILobSimulator(config)
    next_probe = 0.0
    probes: dict[int, tuple[Side, int, int, float]] = {}  # oid -> (side, delta, qa, t0)
    while sim.t < horizon:
        sim.step()
        if sim.t >= next_probe and sim.t <= horizon - min_runway:
            bb, ba = sim.best_bid, sim.best_ask
            for side in checked_sides:
                touch = bb if side == "buy" else ba
                if touch is None:
                    continue
                for d in deltas:
                    price = touch - d * config.tick if side == "buy" else touch + d * config.tick
                    try:
                        oid = sim.submit_limit_order(side, price, tag=PROBE_TAG)
                    except ValueError:
                        continue  # marketable at submit: skip, don't snap
                    qa = sim.queue_ahead_at_submit(oid)
                    probes[oid] = (side, d, int(qa) if qa is not None else -1, sim.t)
            next_probe = sim.t + probe_interval

    filled: dict[int, float] = {}
    for tr in sim.trades:
        if tr.maker_tag == PROBE_TAG and tr.maker_order_id in probes:
            filled[tr.maker_order_id] = tr.t

    records: list[FillRecord] = []
    for oid, (side, d, qa, t0) in probes.items():
        if oid in filled:
            records.append(
                FillRecord(
                    side=side,
                    delta_ticks=d,
                    queue_ahead=qa,
                    t_submit=t0,
                    duration=filled[oid] - t0,
                    event=True,
                )
            )
        else:
            records.append(
                FillRecord(
                    side=side,
                    delta_ticks=d,
                    queue_ahead=qa,
                    t_submit=t0,
                    duration=horizon - t0,
                    event=False,
                )
            )
    return records


def kaplan_meier(records: list[FillRecord]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Product-limit survival curve: returns (times, S) with S(0)=1.

    Ties are resolved by counting all events and censorings at a time
    against the full risk set (standard KM convention).
    """
    if not records:
        raise ValueError("records must be non-empty")
    times = np.unique(np.asarray([r.duration for r in records], dtype=float))
    surv: list[float] = [1.0]
    t_out: list[float] = [0.0]
    s = 1.0
    for t in times:
        at_risk = sum(1 for r in records if r.duration >= t)
        events = sum(1 for r in records if r.event and r.duration == t)
        if at_risk == 0:
            continue
        if events:
            s *= 1.0 - events / at_risk
            t_out.append(float(t))
            surv.append(s)
    return np.asarray(t_out), np.asarray(surv)


def nelson_aalen(records: list[FillRecord]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cumulative hazard Λ(t) = Σ d_i/n_i over distinct event times."""
    if not records:
        raise ValueError("records must be non-empty")
    times = np.unique(np.asarray([r.duration for r in records if r.event], dtype=float))
    lam: list[float] = [0.0]
    t_out: list[float] = [0.0]
    acc = 0.0
    for t in times:
        at_risk = sum(1 for r in records if r.duration >= t)
        events = sum(1 for r in records if r.event and r.duration == t)
        if at_risk == 0:
            continue
        acc += events / at_risk
        t_out.append(float(t))
        lam.append(acc)
    return np.asarray(t_out), np.asarray(lam)


@dataclass(frozen=True)
class ExpIntensityFit:
    """Fitted ``lambda(delta) = A exp(-kappa delta)`` fill intensity."""

    a: float
    kappa: float
    rates: dict[int, float]  # empirical Poisson rate per delta bucket
    exposure: dict[int, float]


def fit_exp_intensity(records: list[FillRecord]) -> ExpIntensityFit:
    """Per-bucket Poisson MLE (events/exposure) + weighted log-linear fit.

    Buckets with zero events are excluded from the log-linear fit (log 0
    is undefined; their information enters only via zero rates), so a
    fully-censored deep bucket does not crash the fit.
    """
    if not records:
        raise ValueError("records must be non-empty")
    deltas = sorted({r.delta_ticks for r in records})
    rates: dict[int, float] = {}
    exposure: dict[int, float] = {}
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    for d in deltas:
        sub = [r for r in records if r.delta_ticks == d]
        events = sum(1 for r in sub if r.event)
        expo = sum(r.duration for r in sub)
        if expo <= 0.0:
            raise ValueError(f"delta bucket {d} has zero exposure")
        exposure[d] = expo
        rates[d] = events / expo
        if events > 0:
            xs.append(float(d))
            ys.append(math.log(rates[d]))
            ws.append(float(events))
    if len(xs) < 2:
        raise ValueError("need >= 2 delta buckets with fills to fit kappa")
    w = np.asarray(ws)
    x = np.asarray(xs)
    y = np.asarray(ys)
    wsum = float(w.sum())
    xbar = float((w * x).sum() / wsum)
    ybar = float((w * y).sum() / wsum)
    cov = float((w * (x - xbar) * (y - ybar)).sum())
    var = float((w * (x - xbar) ** 2).sum())
    if var <= 0.0:
        raise ValueError("delta grid is degenerate")
    kappa = -cov / var  # slope of log rate on delta is -kappa
    a = math.exp(ybar + kappa * xbar)
    return ExpIntensityFit(a=a, kappa=kappa, rates=rates, exposure=exposure)


def fill_prob(fit: ExpIntensityFit, delta_ticks: int, t: float) -> float:
    """P(fill by ``t``) at tick offset ``delta`` under the fitted model."""
    _nonneg_finite(t, "t")
    if int(delta_ticks) < 0:
        raise ValueError(f"delta_ticks must be >= 0, got {delta_ticks!r}")
    lam = fit.a * math.exp(-fit.kappa * int(delta_ticks))
    return 1.0 - math.exp(-lam * t)


def _bucket_cal_error(records: list[FillRecord], fit: ExpIntensityFit, t_eval: float) -> float:
    """Mean |predicted fill prob − empirical fill freq| over records with
    a full ``t_eval`` runway (so empirical frequency is not censored)."""
    errs: list[float] = []
    for d in sorted({r.delta_ticks for r in records}):
        # informative = full runway (duration >= t_eval) or a fill by t_eval
        sub = [r for r in records if r.delta_ticks == d and (r.duration >= t_eval or r.event)]
        if not sub:
            continue
        emp = sum(1 for r in sub if r.event and r.duration <= t_eval) / len(sub)
        errs.append(abs(fill_prob(fit, d, t_eval) - emp))
    return float(np.mean(errs)) if errs else float("nan")


def fill_hazard_bench(
    *,
    n_probe_horizon: float = 3000.0,
    probe_interval: float = 25.0,
    seed: int = 0,
    t_eval: float = 100.0,
) -> dict[str, Any]:
    """Sealed ``fill_hazard.v1`` receipt over one calibration run."""
    from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

    _pos_finite(n_probe_horizon, "n_probe_horizon")
    _pos_finite(t_eval, "t_eval")
    cfg = santa_fe_config(seed=seed)
    records = collect_fill_records(
        cfg,
        horizon=n_probe_horizon,
        probe_interval=probe_interval,
    )
    if not records:
        raise RuntimeError("no probe records collected")
    n_events = sum(1 for r in records if r.event)
    if n_events < 10:
        raise RuntimeError(f"too few fills to estimate hazard ({n_events})")

    fit = fit_exp_intensity(records)
    # KM per bucket: check monotone ordering of median fill times in delta.
    medians: dict[int, float] = {}
    for d in sorted({r.delta_ticks for r in records}):
        sub = [r for r in records if r.delta_ticks == d]
        t_km, s_km = kaplan_meier(sub)
        below = np.nonzero(s_km <= 0.5)[0]
        medians[d] = float(t_km[below[0]]) if below.size else float("inf")
    finite_meds = [m for m in medians.values() if math.isfinite(m)]
    km_monotone = bool(
        len(finite_meds) >= 2
        and all(
            medians[d1] <= medians[d2]
            for d1, d2 in zip(sorted(medians), sorted(medians)[1:], strict=True)
            if math.isfinite(medians[d1]) and math.isfinite(medians[d2])
        )
    )
    cal_err = _bucket_cal_error(records, fit, t_eval)
    censor_frac = 1.0 - n_events / len(records)

    payload: dict[str, Any] = {
        "schema": FILL_HAZARD_SCHEMA,
        "kind": "fill_hazard",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "Probe-order survival on the synthetic ZI-LOB engine; "
            "simulator-internal evidence, never market claims."
        ),
        "n_records": len(records),
        "n_fills": n_events,
        "censor_fraction": float(censor_frac),
        "deltas": sorted({r.delta_ticks for r in records}),
        "bucket_rates": {str(k): v for k, v in fit.rates.items()},
        "median_fill_time_by_delta": {str(k): v for k, v in medians.items()},
        "km_monotone_in_delta": km_monotone,
        "exp_a": float(fit.a),
        "exp_kappa": float(fit.kappa),
        "calibration_mean_abs_err": cal_err,
        "t_eval": float(t_eval),
    }
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "FILL_HAZARD_SCHEMA",
    "PROBE_TAG",
    "ExpIntensityFit",
    "FillRecord",
    "collect_fill_records",
    "fill_hazard_bench",
    "fill_prob",
    "fit_exp_intensity",
    "kaplan_meier",
    "nelson_aalen",
]
