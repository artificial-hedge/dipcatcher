"""Analytic fill-time model: queue depletion as a Gamma race.

The empirical counterpart lives in ``fill_hazard``; this module is the
closed-form version in the Cont & De Larrard spirit: a resting order's
queue ahead ``q`` is a pure-death process — each unit ahead dies
independently at rate ``delta`` (MO sweeps plus cancellations at the
level). The fill time is then the sum of ``q`` iid Exp(delta) waits:

    T_fill ~ Gamma(q, rate=delta)
    mean = q/delta,  var = q/delta**2,
    S(t) = P(T > t) = exp(-delta t) * sum_{k=0}^{q-1} (delta t)^k / k!

(the Poisson-tail form of the gamma survival function — exactly the
statement "fill needs q deaths, each a rate-delta Poisson mark").

- ``death_stats`` — closed-form mean/variance.
- ``fill_cdf`` / ``fill_survival`` — gamma CDF via the Poisson-sum form.
- ``queue_depletion_bench`` — sealed ``queue_depletion.v1``: gamma model
  vs the empirical probe lane on the ZI-LOB (probe orders at the touch,
  records grouped by ``queue_ahead_at_submit``; per-bucket mean fill time
  and max |model CDF - empirical CDF|). The effective depletion rate is
  measured, not assumed: delta_hat = (MO rate hitting the touch +
  cancel rate at the level) per unit of queue depth.

SYNTHETIC only.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_fund.microstructure.zi_lob_simulator import (
    Side,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

QUEUE_DEPLETION_SCHEMA = "queue_depletion.v1"
QD_PROBE_TAG = "qd_probe"


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


def _check_q(q: int) -> int:
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or int(q) < 1:
        raise ValueError(f"queue_ahead must be an int >= 1, got {q!r}")
    return int(q)


def death_stats(q: int, delta: float) -> dict[str, float]:
    """Closed-form mean/variance of Gamma(q, rate=delta) fill time."""
    q = _check_q(q)
    delta = _pos_finite(delta, "delta")
    return {"mean": q / delta, "var": q / (delta * delta), "q": float(q), "delta": delta}


def fill_survival(q: int, delta: float, t: float) -> float:
    """S(t) = exp(-delta t) * sum_{k=0}^{q-1} (delta t)^k / k!."""
    q = _check_q(q)
    delta = _pos_finite(delta, "delta")
    t = _nonneg_finite(t, "t")
    x = delta * t
    # k=0..q-1 Poisson partial sum, computed stably via recursion
    term = 1.0
    s = 1.0
    for k in range(1, q):
        term *= x / k
        s += term
    return float(math.exp(-x) * s)


def fill_cdf(q: int, delta: float, t: float) -> float:
    """P(T_fill <= t) = 1 - S(t)."""
    return 1.0 - fill_survival(q, delta, t)


def fill_quantile(q: int, delta: float, level: float) -> float:
    """Smallest t with F(t) >= level via bisection (monotone in t)."""
    q = _check_q(q)
    delta = _pos_finite(delta, "delta")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0,1), got {level!r}")
    lo, hi = 0.0, 10.0 * death_stats(q, delta)["mean"]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fill_cdf(q, delta, mid) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ProbeOutcome:
    """One touch-probe order: queue ahead at submit + fill/censor time."""

    side: str
    queue_ahead: int
    duration: float
    event: bool


def collect_touch_probes(
    config: ZILobConfig,
    *,
    horizon: float,
    probe_interval: float = 25.0,
    min_runway: float = 10.0,
) -> tuple[list[ProbeOutcome], dict[str, float]]:
    """Probe both touches; also measure the depletion-rate ingredients.

    Returns (records, rates) where rates has ``mo_rate`` (MOs/sec),
    ``cxl_rate_at_touch`` (cancels/sec at the best level, proxied by
    total cancel events per unit of average touch depth).
    """
    _pos_finite(horizon, "horizon")
    _pos_finite(probe_interval, "probe_interval")
    sim = ZILobSimulator(config)
    probes: dict[int, tuple[Side, int, float]] = {}  # oid -> (side, qa, t0)
    next_probe = 0.0
    n_mo = 0
    n_cxl = 0
    while sim.t < horizon:
        kind = sim.step()
        if kind == "market":
            n_mo += 1
        elif kind == "cancel":
            n_cxl += 1
        if sim.t >= next_probe and sim.t <= horizon - min_runway:
            touches: tuple[tuple[Side, float | None], ...] = (
                ("buy", sim.best_bid),
                ("sell", sim.best_ask),
            )
            for side, touch in touches:
                if touch is None:
                    continue
                try:
                    oid = sim.submit_limit_order(side, touch, tag=QD_PROBE_TAG)
                except ValueError:
                    continue
                qa = sim.queue_ahead_at_submit(oid)
                probes[oid] = (side, int(qa) if qa is not None else -1, sim.t)
            next_probe = sim.t + probe_interval

    filled: dict[int, float] = {
        tr.maker_order_id: tr.t for tr in sim.trades if tr.maker_tag == QD_PROBE_TAG
    }
    recs = [
        ProbeOutcome(
            side=s,
            queue_ahead=qa,
            duration=(filled[o] - t0) if o in filled else (horizon - t0),
            event=o in filled,
        )
        for o, (s, qa, t0) in probes.items()
        if qa > 0
    ]
    rates = {"mo_rate": n_mo / horizon, "cxl_rate": n_cxl / horizon}
    return recs, rates


def empirical_fill_stats(records: list[ProbeOutcome]) -> dict[int, dict[str, float]]:
    """Per queue_ahead bucket: mean fill time (fills only) + fill fraction."""
    if not records:
        raise ValueError("records must be non-empty")
    out: dict[int, dict[str, float]] = {}
    by_q: dict[int, list[ProbeOutcome]] = defaultdict(list)
    for r in records:
        by_q[r.queue_ahead].append(r)
    for q, recs in sorted(by_q.items()):
        fills = [r.duration for r in recs if r.event]
        out[q] = {
            "n": float(len(recs)),
            "n_fills": float(len(fills)),
            "mean_fill_time": float(np.mean(fills)) if fills else float("nan"),
            "fill_frac": len(fills) / len(recs),
        }
    return out


def max_cdf_gap(records: list[ProbeOutcome], delta: float, *, t_max: float) -> float:
    """Max over buckets and a time grid of |model CDF - empirical CDF|.

    Empirical CDF uses uncensored records only at t: a record informs t
    if it filled by t (numerator) or survived past t (denominator) —
    censored-before-t records are uninformative there and excluded.
    """
    delta = _pos_finite(delta, "delta")
    t_max = _pos_finite(t_max, "t_max")
    by_q: dict[int, list[ProbeOutcome]] = defaultdict(list)
    for r in records:
        by_q[r.queue_ahead].append(r)
    grid = np.linspace(0.0, t_max, 25)[1:]
    worst = 0.0
    for q, recs in by_q.items():
        for t in grid:
            denom = [r for r in recs if r.duration >= t or r.event]
            if len(denom) < 3:
                continue
            emp = sum(1 for r in denom if r.event and r.duration <= t) / len(denom)
            worst = max(worst, abs(fill_cdf(q, delta, float(t)) - emp))
    return float(worst)


def queue_depletion_bench(
    *,
    horizon: float = 2000.0,
    probe_interval: float = 25.0,
    seed: int = 0,
    t_max: float = 150.0,
) -> dict[str, Any]:
    """Sealed ``queue_depletion.v1`` receipt."""
    from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

    _pos_finite(horizon, "horizon")
    _pos_finite(t_max, "t_max")
    cfg = santa_fe_config(seed=seed)
    recs, rates = collect_touch_probes(cfg, horizon=horizon, probe_interval=probe_interval)
    if len(recs) < 20:
        raise RuntimeError(f"too few probe records ({len(recs)})")
    stats = empirical_fill_stats(recs)
    # Effective per-unit depletion rate: touch-level death events are MO
    # arrivals that hit our side (~half of all MOs) + cancels at the
    # touch. Units are individual orders: delta ~ (0.5*mo_rate + cxl
    # share per unit). Per-unit rate over a level of size ~q is roughly
    # (0.5*mo_rate)/E[q]; estimate empirically: delta = fills /
    # sum(durations) per unit queue position — the hazard form
    # events/exposure normalized by q.
    num = 0.0
    den = 0.0
    for r in recs:
        num += 1.0 if r.event else 0.0
        den += r.duration * r.queue_ahead
    if den <= 0.0:
        raise RuntimeError("zero exposure")
    delta_hat = num / den
    gap = max_cdf_gap(recs, delta_hat, t_max=t_max)
    model_means = {str(q): death_stats(q, delta_hat)["mean"] for q in sorted(stats)}
    rel_errs = {
        str(q): abs(model_means[str(q)] - stats[q]["mean_fill_time"]) / stats[q]["mean_fill_time"]
        for q in stats
        if math.isfinite(stats[q]["mean_fill_time"]) and stats[q]["n_fills"] >= 5
    }
    payload: dict[str, Any] = {
        "schema": QUEUE_DEPLETION_SCHEMA,
        "kind": "queue_depletion",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "Gamma queue-depletion model vs empirical probes on the "
            "synthetic ZI-LOB; never market evidence."
        ),
        "n_probes": len(recs),
        "n_fills": int(sum(1 for r in recs if r.event)),
        "delta_hat": float(delta_hat),
        "mo_rate": float(rates["mo_rate"]),
        "cxl_rate": float(rates["cxl_rate"]),
        "bucket_stats": {str(k): v for k, v in stats.items()},
        "model_mean_fill_time": model_means,
        "max_cdf_gap": float(gap),
        "mean_fill_time_rel_err": {k: float(v) for k, v in rel_errs.items()},
        "mean_rel_err": float(np.mean(list(rel_errs.values()))) if rel_errs else float("nan"),
        "t_max": float(t_max),
    }
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "QD_PROBE_TAG",
    "QUEUE_DEPLETION_SCHEMA",
    "ProbeOutcome",
    "collect_touch_probes",
    "death_stats",
    "empirical_fill_stats",
    "fill_cdf",
    "fill_quantile",
    "fill_survival",
    "max_cdf_gap",
    "queue_depletion_bench",
]
