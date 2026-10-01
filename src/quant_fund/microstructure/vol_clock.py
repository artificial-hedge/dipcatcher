"""vol_clock — transaction-time vs calendar-time volatility on the ZI-LOB.

Easley–O'Hara time deformation: asset prices move on a *business* clock
driven by information arrivals, not on the wall clock. If the hypothesis
holds in this sim, returns sampled every k market-order events should
carry less |return| autocorrelation than returns sampled at fixed Δt —
calendar batches mix quiet and active stretches, inflating measured
volatility clustering; the event clock samples each flow burst at an
even rate.

The bench reports Ljung–Box statistics for |ret| under both clocks in
calm and drifting regimes — an honest mechanism measurement, labeled
SYNTHETIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

VOL_CLOCK_SCHEMA = "vol_clock.v1"


@dataclass(frozen=True)
class ClockTape:
    """Mid marks sampled on the two clocks of the same session."""

    event_rets: NDArray[np.float64]  # per-k-MO log returns
    calendar_rets: NDArray[np.float64]  # per-Δt log returns
    n_mo: int


def acf_abs(ret: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """Autocorrelation of |r| up to ``max_lag``."""
    x = np.abs(ret)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.zeros(max_lag)
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        out[lag - 1] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return out


def ljung_box(acf_vals: NDArray[np.float64], n: int) -> float:
    """Ljung–Box Q for a precomputed ACF: Q = n(n+2) Σ ρ_k²/(n−k)."""
    q = 0.0
    for k, rho in enumerate(acf_vals, start=1):
        if n - k <= 0:
            break
        q += rho * rho / (n - k)
    return float(n * (n + 2) * q)


def collect_tape(
    *,
    config: ZILobConfig,
    horizon: float,
    flow: MarkovRegimeFlow | None = None,
    event_k: int = 5,
    calendar_dt: float = 1.0,
) -> ClockTape:
    """One session, two sampling clocks on the identical tape."""
    sim = ZILobSimulator(config) if flow is None else ZILobSimulator(config, flow=flow)
    event_mids: list[float] = []
    cal_mids: list[float] = []
    n_mo = 0
    next_cal = 0.0
    while sim.t < horizon:
        kind = sim.step()
        if kind == "market":
            n_mo += 1
            if n_mo % event_k == 0:
                m = sim.mid
                if m is not None:
                    event_mids.append(m)
        while sim.t >= next_cal:
            m = sim.mid
            if m is not None:
                cal_mids.append(m)
            next_cal += calendar_dt
    er = np.diff(np.log(np.asarray(event_mids))) if len(event_mids) > 1 else np.empty(0)
    cr = np.diff(np.log(np.asarray(cal_mids))) if len(cal_mids) > 1 else np.empty(0)
    return ClockTape(event_rets=er, calendar_rets=cr, n_mo=n_mo)


def _flow(seed: int, p_buy_trend: float = 0.78) -> MarkovRegimeFlow:
    return MarkovRegimeFlow(
        states=[
            RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
            RegimeState(name="trend_buy", intensity_mult=1.5, p_buy=p_buy_trend),
        ],
        stay_probs=[0.97, 0.94],
        seed=seed,
    )


def vol_clock_bench(
    *,
    n_seeds: int = 10,
    horizon: float = 600.0,
    max_lag: int = 15,
) -> dict[str, Any]:
    """LB(|r|) on both clocks, calm vs drift — sealed payload."""
    arms: dict[str, dict[str, list[float]]] = {
        "calm": {"event": [], "calendar": []},
        "drift": {"event": [], "calendar": []},
    }
    for k in range(n_seeds):
        calm = collect_tape(
            config=ZILobConfig(seed=9000 + k, init_depth=8, band=8),
            horizon=horizon,
        )
        drift = collect_tape(
            config=ZILobConfig(seed=9000 + k, init_depth=8, band=8),
            horizon=horizon,
            flow=_flow(9000 + k),
        )
        for arm, tape in (("calm", calm), ("drift", drift)):
            arms[arm]["event"].append(
                ljung_box(acf_abs(tape.event_rets, max_lag), tape.event_rets.size)
                if tape.event_rets.size > max_lag + 5
                else float("nan")
            )
            arms[arm]["calendar"].append(
                ljung_box(acf_abs(tape.calendar_rets, max_lag), tape.calendar_rets.size)
                if tape.calendar_rets.size > max_lag + 5
                else float("nan")
            )

    def mean_ok(xs: list[float]) -> float:
        a = np.asarray(xs)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    out: dict[str, dict[str, float]] = {}
    for arm, d in arms.items():
        out[arm] = {
            "lb_event_mean": mean_ok(d["event"]),
            "lb_calendar_mean": mean_ok(d["calendar"]),
            "ratio_event_over_calendar": (
                mean_ok(d["event"]) / mean_ok(d["calendar"])
                if mean_ok(d["calendar"]) > 0
                else float("nan")
            ),
        }
    payload: dict[str, Any] = {
        "schema": VOL_CLOCK_SCHEMA,
        "kind": "vol_clock",
        "n_seeds": n_seeds,
        "horizon": horizon,
        "max_lag": max_lag,
        "arms": out,
        "event_clock_less_autocorrelated": bool(
            np.isfinite(out["drift"]["ratio_event_over_calendar"])
            and out["drift"]["ratio_event_over_calendar"] < 1.0
        ),
        "interpretation": (
            "Ljung-Box on |r| under both clocks, both regimes; ratio<1 in "
            "drift means event sampling sees less volatility clustering — "
            "the transaction-time hypothesis is measured, not assumed"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
