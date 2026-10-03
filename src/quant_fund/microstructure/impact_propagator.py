"""Transient-impact propagator estimation on the ZI-LOB engine.

Gatheral's propagator model says a market order's mid-price impact decays
as ``G(tau) ~ tau^{-theta}``. This module measures the signed response
function on the synthetic engine directly:

    R(tau) = E[ eps_t * (mid_{t+tau} - mid_t) ]

over all market-order events, where ``eps_t`` is the MO direction
(+1 buy, -1 sell). R(0+) is the instantaneous impact; its decay rate
``theta`` is fitted by weighted log-log regression on lags.

- ``mo_response_series`` — runs the sim, records every MO's direction
  and the mid path; computes R(tau) at a lag grid in *event* units
  (post-event mid measured at the k-th subsequent event timestamp —
  event-time avoids nonuniform wall-clock interpolation).
- ``fit_power_law`` — log-log OLS on positive-response lags; returns
  theta, intercept, R^2; fails closed on degenerate fits.
- ``sign_shuffled_control`` — same estimator with randomized eps;
  must give ~zero response (negative control on estimator bias).
- ``impact_propagator_bench`` — sealed ``impact_propagator.v1`` receipt:
  response curve, fitted decay, control magnitude.

SYNTHETIC only — the ZI engine has no latent impact model, so any
measured propagation is purely book-depletion + reversion mechanics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

IMPACT_PROP_SCHEMA = "impact_propagator.v1"


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


@dataclass(frozen=True)
class ResponseCurve:
    """Signed response ``R(k)`` at event-lag ``k`` (k = 0 is the MO itself)."""

    lags: NDArray[np.float64]
    response: NDArray[np.float64]  # tick units
    n_events: int
    counts: NDArray[np.int64]  # pairs contributing per lag


def mo_response_series(
    config: ZILobConfig,
    *,
    horizon: float,
    max_lag: int = 50,
) -> ResponseCurve:
    """Measure R(k) over the sim's market-order events.

    Every ``"market"`` step that produced a trade is an impulse
    (``tr.maker_tag`` names the *resting* side's tag, not the aggressor's
    — filtering on it would wrongly drop events). Mid is recorded after
    every event, so R(k) uses the k-th post-MO event's mid — event-time
    lag. R(0) is the instantaneous impact.
    """
    _pos_finite(horizon, "horizon")
    if isinstance(max_lag, bool) or int(max_lag) < 1:
        raise ValueError(f"max_lag must be an int >= 1, got {max_lag!r}")
    sim = ZILobSimulator(config)
    eps: list[float] = []
    all_mids: list[float] = []
    mo_idx: list[int] = []
    while sim.t < horizon:
        n_before = len(sim.trades)
        kind = sim.step()
        mid = sim.mid
        if mid is None:
            all_mids.append(float("nan"))
            continue
        all_mids.append(float(mid))
        if kind == "market" and len(sim.trades) > n_before:
            tr = sim.trades[-1]
            if tr.aggressor not in ("buy", "sell"):
                continue
            mo_idx.append(len(all_mids) - 1)
            eps.append(1.0 if tr.aggressor == "buy" else -1.0)
    return _response_from_series(
        np.asarray(all_mids), mo_idx, np.asarray(eps), max_lag, config.tick
    )


def _response_from_series(
    mids: NDArray[np.float64],
    mo_idx: list[int],
    eps: NDArray[np.float64],
    max_lag: int,
    tick: float,
) -> ResponseCurve:
    lags = np.arange(0, max_lag + 1, dtype=float)
    resp = np.full(lags.shape, np.nan)
    counts = np.zeros(lags.shape, dtype=np.int64)
    for k in range(max_lag + 1):
        acc = 0.0
        n = 0
        for i, mi in enumerate(mo_idx):
            j = mi + k
            if j >= mids.size or not math.isfinite(mids[j]):
                continue
            # mid at the MO event is mid *after* the trade consumed the
            # touch; R(0) = 0 by construction, so we compare against the
            # mid just before the MO (previous event mid).
            if mi == 0 or not math.isfinite(mids[mi - 1]):
                continue
            acc += eps[i] * (mids[j] - mids[mi - 1])
            n += 1
        if n:
            resp[k] = acc / n
            counts[k] = n
    return ResponseCurve(lags=lags, response=resp / tick, n_events=len(mo_idx), counts=counts)


def fit_power_law(curve: ResponseCurve) -> dict[str, float]:
    """Weighted log-log fit ``R(k) = a k^{-theta}`` on lags k >= 1 with
    positive response; weights = sample counts (larger lags noisier)."""
    if not isinstance(curve, ResponseCurve):
        raise TypeError("curve must be a ResponseCurve")
    lags, resp, counts = curve.lags, curve.response, curve.counts
    mask = (lags >= 1.0) & np.isfinite(resp) & (resp > 0.0) & (counts >= 5)
    if int(mask.sum()) < 3:
        raise ValueError("need >= 3 positive-response lags with >= 5 pairs")
    x = np.log(lags[mask])
    y = np.log(resp[mask])
    w = counts[mask].astype(float)
    wsum = w.sum()
    xb = float((w * x).sum() / wsum)
    yb = float((w * y).sum() / wsum)
    sxx = float((w * (x - xb) ** 2).sum())
    if sxx <= 0.0:
        raise ValueError("degenerate lag grid")
    slope = float((w * (x - xb) * (y - yb)).sum() / sxx)
    intercept = yb - slope * xb
    resid = y - (intercept + slope * x)
    tss = float((w * (y - yb) ** 2).sum())
    rss = float((w * resid**2).sum())
    r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
    return {"theta": -slope, "a": math.exp(intercept), "r2": r2, "n_lags": float(mask.sum())}


def sign_shuffled_control(
    config: ZILobConfig, *, horizon: float, max_lag: int = 50, seed: int = 0
) -> ResponseCurve:
    """Same estimator, MO directions randomly permuted — must be ~flat."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    # Recompute the event/mid series, then permute signs
    base = _mo_series(config, horizon)
    rng = np.random.default_rng(seed)
    eps = base["eps"].copy()
    rng.shuffle(eps)
    return _response_from_series(base["mids"], base["idx"], eps, max_lag, config.tick)


def _mo_series(config: ZILobConfig, horizon: float) -> dict[str, Any]:
    sim = ZILobSimulator(config)
    all_mids: list[float] = []
    mo_idx: list[int] = []
    eps: list[float] = []
    while sim.t < horizon:
        n_before = len(sim.trades)
        kind = sim.step()
        mid = sim.mid
        if mid is None:
            all_mids.append(float("nan"))
            continue
        all_mids.append(float(mid))
        if kind == "market" and len(sim.trades) > n_before:
            tr = sim.trades[-1]
            mo_idx.append(len(all_mids) - 1)
            eps.append(1.0 if tr.aggressor == "buy" else -1.0)
    return {
        "mids": np.asarray(all_mids),
        "idx": mo_idx,
        "eps": np.asarray(eps),
    }


def impact_propagator_bench(
    *,
    horizon: float = 2000.0,
    max_lag: int = 40,
    seed: int = 0,
) -> dict[str, Any]:
    """Sealed ``impact_propagator.v1`` receipt."""
    from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

    _pos_finite(horizon, "horizon")
    cfg = santa_fe_config(seed=seed)
    curve = mo_response_series(cfg, horizon=horizon, max_lag=max_lag)
    if curve.n_events < 30:
        raise RuntimeError(f"too few MO events ({curve.n_events})")
    fit_error: str | None = None
    try:
        fit = fit_power_law(curve)
    except ValueError as exc:
        fit_error = str(exc)
        fit = {
            "theta": float("nan"),
            "a": float("nan"),
            "r2": float("nan"),
            "n_lags": 0.0,
        }
    ctrl = sign_shuffled_control(cfg, horizon=horizon, max_lag=max_lag, seed=seed)
    r_finite = curve.response[np.isfinite(curve.response)]
    c_finite = ctrl.response[np.isfinite(ctrl.response)]
    payload: dict[str, Any] = {
        "schema": IMPACT_PROP_SCHEMA,
        "kind": "impact_propagator",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "Event-time response function on the synthetic ZI-LOB; "
            "impact propagation here is book mechanics, never market evidence."
        ),
        "n_mo_events": curve.n_events,
        "max_lag": int(max_lag),
        "response_ticks": np.nan_to_num(curve.response, nan=0.0).tolist(),
        "counts": curve.counts.tolist(),
        "max_abs_response_ticks": float(np.abs(r_finite).max()) if r_finite.size else float("nan"),
        "theta_hat": float(fit["theta"]),
        "prefactor_a": float(fit["a"]),
        "r2": float(fit["r2"]),
        "control_max_abs_ticks": float(np.abs(c_finite).max()) if c_finite.size else float("nan"),
        "fit_error": fit_error,
    }
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "IMPACT_PROP_SCHEMA",
    "ResponseCurve",
    "fit_power_law",
    "impact_propagator_bench",
    "mo_response_series",
    "sign_shuffled_control",
]
