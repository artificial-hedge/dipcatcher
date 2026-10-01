"""Inter-event / inter-execution waiting-time analysis on the ZI-LOB sim.

**Labeled SYNTHETIC** research diagnostics (lane B4-i companion): the
distribution of durations between successive engine events and between
successive executions is measured inside
``microstructure.zi_lob_simulator`` — the zero-intelligence LOB whose event
clock is a superposition of state-dependent Poisson rates — under two flow
arms:

- ``iid``: no modulating flow (``flow=None``); market-order arrivals form a
  homogeneous Poisson stream, so inter-execution durations should be
  near-exponential (burstiness ~ 0, CV ~ 1, Weibull shape ~ 1).
- ``regime``: ``MarkovRegimeFlow`` calm/bursty two-state modulation on the
  MO clock (Moret & Lillo 2026, Sec. 5); the mixture of fast and slow
  clocks produces heavy-tailed waits — positive burstiness, CV > 1,
  Weibull shape < 1.

Per arm and per stream (all events / executions only) the module reports:
(a) the empirical inter-execution duration histogram next to the expected
counts under the maximum-likelihood exponential fit, summarized by the
two-sided Kolmogorov-Smirnov statistic between the empirical CDF and the
fitted exponential CDF; (b) the Goh & Barabási (2008) burstiness parameter
``B = (sigma - mu) / (sigma + mu)`` (−1 periodic, 0 Poisson, → 1 bursty);
(c) the coefficient of variation ``sigma / mu``; (d) the Weibull shape from
a log-log survival regression (``log(-log S(t)) = k*log t − k*log lam``
with the ``S_(i) = 1 − i/(n+1)`` plotting position).

Fail-closed: an arm with fewer than ``MIN_EXECUTIONS`` executions raises —
duration statistics on a handful of fills are noise, not evidence. All
outputs are SYNTHETIC correctness diagnostics, never market evidence; the
bundle seals via ``receipt_sha256`` over the canonical JSON body (the
``canonical_json`` convention of ``research.receipt_v2._seal_errors``).

References:
- Goh, Barabási (2008). Burstiness and memory in complex systems.
  *EPL* 81:48002 — burstiness parameter B.
- Cont, Stoikov, Talreja (2010). A stochastic model for order book
  dynamics. *Operations Research* 58(1):191-205 — ZI-LOB queueing.
- Moret, Lillo (2026). arXiv:2609.11614 — regime-switching MO flow on the
  MO clock.
- Clauset, Shalizi, Newman (2009). *SIAM Review* 51(4):661-703 — tail-fit
  regression hygiene.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    ZI_LOB_REVISION,
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

Array = NDArray[np.float64]

MIN_EXECUTIONS = 50
MIN_DURATIONS = 30
WAITING_TIMES_REVISION = "SYNTHETIC_WAITING_TIMES_v1"

#: Calm/bursty regimes for the ``regime`` arm: the calm clock ticks MOs at
#: half intensity, the bursty clock at 6x; the chain is sticky on the MO
#: clock so bursts arrive as physical-time clusters separated by long gaps.
REGIME_STATES = (
    RegimeState("calm", intensity_mult=0.5, p_buy=0.5),
    RegimeState("bursty", intensity_mult=6.0, p_buy=0.5),
)
REGIME_STAY_PROBS = (0.95, 0.90)


# ---------------------------------------------------------------------------
# Fail-closed validation helpers
# ---------------------------------------------------------------------------


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _pos_int(x: int, name: str) -> int:
    if isinstance(x, bool) or int(x) < 1:
        raise ValueError(f"{name} must be an int >= 1, got {x!r}")
    return int(x)


def _as_positive_durations(
    durations: Sequence[float] | Array,
    *,
    min_n: int,
    name: str = "durations",
) -> Array:
    d = np.asarray(durations, dtype=np.float64).ravel()
    if d.size < min_n:
        raise ValueError(f"{name} must contain at least {min_n} entries, got {d.size}")
    if not np.all(np.isfinite(d)):
        raise ValueError(f"{name} must be finite")
    if np.any(d <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return d


# ---------------------------------------------------------------------------
# Scalar duration statistics
# ---------------------------------------------------------------------------


def inter_durations(times: Sequence[float] | Array) -> Array:
    """Successive differences of a strictly increasing event clock."""
    t = np.asarray(times, dtype=np.float64).ravel()
    if t.size < 2:
        raise ValueError(f"times must contain at least 2 entries, got {t.size}")
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")
    d = np.diff(t)
    if np.any(d <= 0.0):
        raise ValueError("times must be strictly increasing")
    return d


def burstiness(durations: Sequence[float] | Array) -> float:
    """Goh & Barabási (2008) burstiness ``B = (sigma - mu)/(sigma + mu)``.

    ``B = -1`` for a periodic (constant-gap) stream, ``0`` for a Poisson
    stream, ``→ 1`` for a bursty one. Uses the population standard
    deviation, matching the original definition.
    """
    d = _as_positive_durations(durations, min_n=2)
    mu = float(d.mean())
    sigma = float(d.std())
    denom = sigma + mu
    if denom <= 0.0:  # pragma: no cover - all-positive durations
        raise ValueError("burstiness undefined when sigma + mu == 0")
    return float((sigma - mu) / denom)


def coefficient_of_variation(durations: Sequence[float] | Array) -> float:
    """``sigma / mu`` of the durations (1.0 for a Poisson stream)."""
    d = _as_positive_durations(durations, min_n=2)
    mu = float(d.mean())
    if mu <= 0.0:  # pragma: no cover - all-positive durations
        raise ValueError("coefficient of variation undefined for zero mean")
    return float(float(d.std()) / mu)


def exponential_ks_statistic(durations: Sequence[float] | Array) -> float:
    """Two-sided KS distance between the empirical CDF and the fitted exponential.

    The exponential rate is the MLE ``lam_hat = 1 / mean``; the statistic is
    ``sup_t |F_emp(t) - (1 - exp(-lam_hat t))|`` over the order statistics.
    """
    d = _as_positive_durations(durations, min_n=MIN_DURATIONS)
    xs = np.sort(d)
    n = xs.size
    lam_hat = 1.0 / float(xs.mean())
    fitted = 1.0 - np.exp(-lam_hat * xs)
    upper = np.arange(1, n + 1, dtype=np.float64) / n
    lower = np.arange(0, n, dtype=np.float64) / n
    return float(np.max(np.maximum(upper - fitted, fitted - lower)))


def weibull_shape_loglog(durations: Sequence[float] | Array) -> tuple[float, float, float]:
    """Weibull shape via log-log survival regression.

    For ``S(t) = exp(-(t/lam)**k)``, ``log(-log S(t)) = k*log t - k*log lam``;
    the empirical survival uses the ``S_(i) = 1 - i/(n+1)`` plotting
    position, and an OLS fit of ``log(-log S)`` on ``log t`` recovers the
    shape ``k`` and scale ``lam``. ``k = 1`` is exponential, ``k < 1`` is
    heavy-tailed (bursty), ``k > 1`` is thin-tailed. Returns
    ``(shape, scale, r_squared)``; fails closed on constant durations where
    the regression is degenerate.
    """
    d = _as_positive_durations(durations, min_n=MIN_DURATIONS)
    if float(np.ptp(d)) <= 0.0:
        raise ValueError("Weibull log-log fit needs non-constant durations")
    xs = np.sort(d)
    n = xs.size
    surv = 1.0 - np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    y = np.log(-np.log(surv))
    x = np.log(xs)
    slope, intercept = np.polyfit(x, y, 1)
    k = float(slope)
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"Weibull log-log fit degenerate (shape {k!r})")
    scale = float(math.exp(-float(intercept) / k))
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    if not (math.isfinite(scale) and math.isfinite(r2)):
        raise ValueError("Weibull log-log fit produced non-finite parameters")
    return k, scale, float(r2)


def duration_histogram(
    durations: Sequence[float] | Array,
    *,
    n_bins: int = 12,
) -> dict[str, Any]:
    """Uniform-bin histogram plus expected counts under the MLE exponential.

    ``expected_exponential_counts[i]`` is
    ``n * (exp(-lam*e_i) - exp(-lam*e_{i+1}))`` for bin ``[e_i, e_{i+1}]`` —
    the expected count had the stream been truly Poisson.
    """
    nb = _pos_int(n_bins, "n_bins")
    d = _as_positive_durations(durations, min_n=2)
    edges = np.linspace(0.0, float(d.max()), nb + 1)
    counts, edges_out = np.histogram(d, bins=edges)
    lam = 1.0 / float(d.mean())
    expected = d.size * (np.exp(-lam * edges[:-1]) - np.exp(-lam * edges[1:]))
    return {
        "bin_edges": [float(e) for e in edges_out],
        "counts": [int(c) for c in counts],
        "expected_exponential_counts": [float(e) for e in expected],
        "exp_rate_hat": float(lam),
        "n_bins": nb,
    }


def duration_stats(
    durations: Sequence[float] | Array,
    *,
    min_n: int = MIN_DURATIONS,
    n_bins: int = 12,
) -> dict[str, Any]:
    """Full per-stream duration summary: histogram, KS, burstiness, CV, Weibull."""
    mn = _pos_int(min_n, "min_n")
    d = _as_positive_durations(durations, min_n=mn)
    hist = duration_histogram(d, n_bins=n_bins)
    k, scale, r2 = weibull_shape_loglog(d)
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "std": float(d.std()),
        "min": float(d.min()),
        "max": float(d.max()),
        "burstiness": burstiness(d),
        "coefficient_of_variation": coefficient_of_variation(d),
        "exp_rate_hat": float(hist["exp_rate_hat"]),
        "ks_stat_exponential": exponential_ks_statistic(d),
        "weibull_shape": k,
        "weibull_scale": scale,
        "weibull_loglog_r2": r2,
        "histogram": hist,
    }


# ---------------------------------------------------------------------------
# Flow arms on the simulator
# ---------------------------------------------------------------------------


def regime_burst_flow(
    *,
    seed: int = 0,
    states: Sequence[RegimeState] = REGIME_STATES,
    stay_probs: Sequence[float] = REGIME_STAY_PROBS,
) -> MarkovRegimeFlow:
    """Calm/bursty two-state MO-clock flow used by the ``regime`` arm."""
    return MarkovRegimeFlow(states, stay_probs, seed=seed)


def run_waiting_time_arm(
    *,
    name: str,
    config: ZILobConfig,
    flow: MarkovRegimeFlow | None,
    horizon: float,
    min_execs: int = MIN_EXECUTIONS,
    min_durations: int = MIN_DURATIONS,
    n_bins: int = 12,
    flow_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one flow arm and measure inter-event / inter-exec durations.

    Steps the simulator to ``horizon`` (continuous seconds on ``sim._t``),
    records every event time and every execution time, and fails closed when
    fewer than ``min_execs`` executions were produced. ``flow_spec`` is an
    optional caller-supplied description of the modulating flow, echoed into
    the report for provenance.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    h = _pos_finite(horizon, "horizon")
    me = _pos_int(min_execs, "min_execs")
    md = _pos_int(min_durations, "min_durations")
    sim = ZILobSimulator(config, flow=flow)
    event_times: list[float] = []
    while sim.t < h:
        sim.step()
        event_times.append(sim.t)
    exec_times = np.asarray([tr.t for tr in sim.trades], dtype=np.float64)
    if exec_times.size < me:
        raise ValueError(
            f"arm {name!r} produced {exec_times.size} execs over {h}s; need >= {me} (fail-closed)"
        )
    event_d = inter_durations(np.asarray(event_times, dtype=np.float64))
    exec_d = inter_durations(exec_times)
    regime_info: dict[str, Any] | None = None
    if flow is not None:
        regime_info = {
            "n_mo": int(flow.n_mo),
            "state_mo_counts": [int(c) for c in flow.state_mo_counts],
            "n_transitions": len(flow.transitions),
            "expected_p_buy": float(flow.expected_p_buy()),
        }
    return {
        "arm": name,
        "flow": "none" if flow is None else "markov_regime",
        "flow_spec": dict(flow_spec) if flow_spec is not None else None,
        "seed": int(config.seed),
        "horizon_seconds": h,
        "elapsed_seconds": float(sim.t),
        "n_events": int(sim.n_events),
        "n_mo_arrivals": int(sim.n_mo_arrivals),
        "n_mo_noop": int(sim.n_mo_noop),
        "n_execs": int(exec_times.size),
        "inter_event": duration_stats(event_d, min_n=md, n_bins=n_bins),
        "inter_exec": duration_stats(exec_d, min_n=md, n_bins=n_bins),
        "regime": regime_info,
        "label": "SYNTHETIC",
    }


def _regime_flow_spec(
    states: Sequence[RegimeState],
    stay_probs: Sequence[float],
) -> dict[str, Any]:
    return {
        "kind": "markov_regime",
        "states": [
            {
                "name": s.name,
                "intensity_mult": float(s.intensity_mult),
                "p_buy": float(s.p_buy),
            }
            for s in states
        ],
        "stay_probs": [float(p) for p in stay_probs],
        "clock": "mo",
    }


def seal_waiting_time_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Stamp ``receipt_sha256`` over the canonical JSON of the payload body.

    The digest is ``hash_bytes(canonical_json_bytes(body))`` where ``body``
    is the payload with the ``receipt_sha256`` field removed — the
    ``canonical_json`` convention of ``research.receipt_v2._seal_errors``.
    """
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    return {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}


def run_waiting_time_arms(
    *,
    seed: int = 0,
    horizon: float = 2000.0,
    min_execs: int = MIN_EXECUTIONS,
    config: ZILobConfig | None = None,
) -> dict[str, Any]:
    """Run the ``iid`` + ``regime`` arms on one seed; sealed SYNTHETIC receipt.

    Both arms share the same ``config`` (common random numbers — the regime
    arm differs only through its modulating flow). The regime arm uses
    ``regime_burst_flow``'s calm/bursty defaults.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    cfg = config if config is not None else ZILobConfig(seed=seed)
    h = _pos_finite(horizon, "horizon")
    states, stay_probs = REGIME_STATES, REGIME_STAY_PROBS
    flow = regime_burst_flow(seed=seed + 1, states=states, stay_probs=stay_probs)
    arms = {
        "iid": run_waiting_time_arm(
            name="iid",
            config=cfg,
            flow=None,
            horizon=h,
            min_execs=min_execs,
            flow_spec={"kind": "none"},
        ),
        "regime": run_waiting_time_arm(
            name="regime",
            config=cfg,
            flow=flow,
            horizon=h,
            min_execs=min_execs,
            flow_spec=_regime_flow_spec(states, stay_probs),
        ),
    }
    payload: dict[str, Any] = {
        "family": "zi_lob_waiting_times",
        "revision": WAITING_TIMES_REVISION,
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "seed": int(seed),
        "horizon_seconds": h,
        "min_execs": int(min_execs),
        "arms": arms,
        "n_arms": len(arms),
    }
    return seal_waiting_time_receipt(payload)
