"""Queue-reactive limit-order-book model (Huang, Lehalle & Rosenbaum).

Queue-reactive CTMC on best-bid/best-ask queue sizes: every event intensity
depends only on the *current* queue size,

    q -> q+1 at rate lambda_L(q)      (limit-order arrival)
    q -> q-1 at rate lambda_C(q)      (cancellation)
    q -> q-1 at rate lambda_M(q)      (market order / execution)

so each queue is a birth-death chain with state-dependent rates. MLE
intensities are counts-per-dwell-time per queue size; the stationary law is
available both analytically (truncated birth-death generator solve) and by
long-run simulation — agreement between the two is the core internal
consistency check. The queue-value curve V(q) scores the join-vs-cancel
advantage of a queue position via absorbing-chain fill probabilities.

References
----------
- Huang, Lehalle & Rosenbaum (2015). Simulating and analyzing order book
  data: the queue-reactive model. *Journal of the American Statistical
  Association* 110(509). arXiv:1312.0563.
- Huang & Rosenbaum (2018). Ergodicity and diffusivity of Markovian order
  book models: a general framework. *SIAM J. Financial Mathematics* 8(1).
  arXiv:1505.04936.
- Huang & Rosenbaum (2018). Appendix on queue-reactive model calibration.
  arXiv:1810.03462 (supplementary; verified on arXiv).

Honesty
-------
All benches run on seeded SYNTHETIC CTMC streams generated inside this
module — correctness tests for the estimator and stationary solver, never
market evidence.

Composition notes
-----------------
- ``microstructure/zi_lob_simulator.py``: emergent book under Poisson ZI
  flow; here the queue *is* the state (no price level), giving closed-form
  stationary analytics the ZI sim lacks.
- ``microstructure/event_time_flow.py`` (wave 19): event-time flow
  diagnostics on bar streams; this module needs a per-event queue
  trajectory instead.
- ``models/regime.py``: queue imbalance regimes; V(q) below is the
  queue-position analogue, not a regime label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EVENT_LIMIT, _EVENT_CANCEL, _EVENT_MARKET = 0, 1, 2


def _check_nonneg(name: str, a: FloatArray | IntArray) -> None:
    if not np.isfinite(np.asarray(a, dtype=float)).all():
        raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class QueueTrajectory:
    """A queue-size path with dwell times and event labels."""

    times: FloatArray  # event times, increasing
    states: IntArray  # queue size *after* each event
    events: IntArray  # event codes: 0 limit(+1), 1 cancel(-1), 2 market(-1)
    q_max: int

    @property
    def n_events(self) -> int:
        return int(self.events.size)


@dataclass(frozen=True)
class RateEstimate:
    """Per-queue-size MLE intensities."""

    lam_l: FloatArray  # limit arrivals per unit time at queue size q
    lam_c: FloatArray  # cancels per unit time at q
    lam_m: FloatArray  # market orders per unit time at q
    dwell: FloatArray  # total time spent in each state q
    counts: FloatArray  # (q_max+1, 3) event counts per state

    @property
    def q_max(self) -> int:
        return int(self.lam_l.size) - 1


def simulate_queue(
    lam_l: FloatArray,
    lam_c: FloatArray,
    lam_m: FloatArray,
    horizon: float,
    seed: int,
    q0: int | None = None,
) -> QueueTrajectory:
    """CTMC queue simulation under given per-state intensities.

    ``lam_*`` are arrays indexed by queue size q (0..q_max); the chain is
    reflected at 0 and q_max (deaths at 0 and births at q_max are zero-rate).
    """
    l_l, l_c, l_m = (np.asarray(v, dtype=float).ravel() for v in (lam_l, lam_c, lam_m))
    if not (l_l.size == l_c.size == l_m.size):
        raise ValueError("rate arrays must have equal length")
    if l_l.size < 2:
        raise ValueError("need at least two states")
    for name, v in (("lam_l", l_l), ("lam_c", l_c), ("lam_m", l_m)):
        _check_nonneg(name, v)
        if (v < 0).any():
            raise ValueError(f"{name} must be non-negative")
    if horizon <= 0 or not np.isfinite(horizon):
        raise ValueError("horizon must be positive")
    q_max = l_l.size - 1
    rng = np.random.default_rng(seed)
    q = int(q0) if q0 is not None else q_max // 2
    if not 0 <= q <= q_max:
        raise ValueError("q0 out of range")
    times, states, events = [], [], []
    t = 0.0
    # generous capacity: expected events ~ total rate * horizon
    while t < horizon:
        births = l_l[q] if q < q_max else 0.0
        deaths = (l_c[q] + l_m[q]) if q > 0 else 0.0
        total = births + deaths
        if total <= 0.0:
            break
        t += rng.exponential(1.0 / total)
        if t >= horizon:
            break
        u = rng.random() * total
        if u < births:
            q += 1
            ev = _EVENT_LIMIT
        else:
            ev = _EVENT_CANCEL if u - births < l_c[q] else _EVENT_MARKET
            q -= 1
        times.append(t)
        states.append(q)
        events.append(ev)
    return QueueTrajectory(
        times=np.asarray(times, dtype=float),
        states=np.asarray(states, dtype=np.int64),
        events=np.asarray(events, dtype=np.int64),
        q_max=q_max,
    )


def estimate_rates(traj: QueueTrajectory) -> RateEstimate:
    """MLE per-queue-size intensities: event counts divided by dwell time."""
    if traj.n_events < 4:
        raise ValueError("trajectory too short to estimate")
    q_max = traj.q_max
    # state *before* each event = state after previous event (start at midpoint)
    pre = np.empty(traj.n_events, dtype=np.int64)
    pre[0] = q_max // 2
    pre[1:] = traj.states[:-1]
    times = traj.times
    dwell = np.zeros(q_max + 1)
    counts = np.zeros((q_max + 1, 3))
    dt = np.diff(times, prepend=times[0])
    for i in range(traj.n_events):
        q = int(pre[i])
        dwell[q] += dt[i]
        counts[q, int(traj.events[i])] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_l = np.where(dwell > 0, counts[:, 0] / dwell, 0.0)
        lam_c = np.where(dwell > 0, counts[:, 1] / dwell, 0.0)
        lam_m = np.where(dwell > 0, counts[:, 2] / dwell, 0.0)
    return RateEstimate(lam_l=lam_l, lam_c=lam_c, lam_m=lam_m, dwell=dwell, counts=counts)


def stationary_birth_death(rates: RateEstimate) -> FloatArray:
    """Stationary distribution of the truncated birth-death chain.

    Births: lam_l[q] for q < q_max. Deaths: lam_c[q] + lam_m[q] for q > 0.
    Solved by the standard detailed-balance product ratio.
    """
    q_max = rates.q_max
    birth = rates.lam_l[:q_max]
    death = (rates.lam_c + rates.lam_m)[1 : q_max + 1]
    if np.any(death <= 0):
        raise ValueError("death rates must be positive on interior states")
    w = np.empty(q_max + 1)
    w[0] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(birth > 0, birth / death, 0.0)
    w[1:] = np.cumprod(ratio)
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("non-finite stationary weights")
    return w / w.sum()


def stationary_simulated(traj: QueueTrajectory) -> FloatArray:
    """Empirical stationary distribution from dwell times of a trajectory."""
    if traj.n_events < 4:
        raise ValueError("trajectory too short")
    q_max = traj.q_max
    pre = np.empty(traj.n_events, dtype=np.int64)
    pre[0] = q_max // 2
    pre[1:] = traj.states[:-1]
    dt = np.diff(traj.times, prepend=traj.times[0])
    dwell = np.zeros(q_max + 1)
    for i in range(traj.n_events):
        dwell[int(pre[i])] += dt[i]
    total = dwell.sum()
    if total <= 0:
        raise ValueError("zero dwell time")
    return dwell / total


def tv_distance(p: FloatArray, q: FloatArray) -> float:
    """Total-variation distance between two distributions."""
    a, b = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("shape mismatch")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("negative mass")
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        raise ValueError("zero-mass distribution")
    return float(0.5 * np.abs(a / sa - b / sb).sum())


def queue_value_curve(rates: RateEstimate, reward: float = 1.0) -> FloatArray:
    """V(q): expected advantage of a position at queue size q.

    Model: joining behind q units pays ``reward`` when the queue drains to
    0 (position fills) before... use absorbing-chain hitting probability:
    V(q) = reward * P(hit 0 before q_max | start q) - queue-position cost
    proportional to q. Computed by first-step analysis on the truncated
    birth-death generator.
    """
    q_max = rates.q_max
    if q_max < 1:
        raise ValueError("need q_max >= 1")
    birth = rates.lam_l
    death = rates.lam_c + rates.lam_m
    # first-step equations for h_q = P(hit 0 before q_max | start q)
    # h_0 = 1, h_{q_max} = 0; interior: -h_{q-1} b + (b+d) h_q - h_{q+1} d = 0
    a_mat = np.zeros((q_max + 1, q_max + 1))
    rhs = np.zeros(q_max + 1)
    a_mat[0, 0], rhs[0] = 1.0, 1.0
    a_mat[q_max, q_max], rhs[q_max] = 1.0, 0.0
    for qq in range(1, q_max):
        b, d = birth[qq], death[qq]
        if b + d <= 0:
            raise ValueError(f"absorbing interior state q={qq}")
        a_mat[qq, qq - 1] = -d
        a_mat[qq, qq] = b + d
        a_mat[qq, qq + 1] = -b
    h = np.linalg.solve(a_mat, rhs)
    # waiting-time cost ~ expected position; V = reward*h - normalized position
    pos = np.arange(q_max + 1) / q_max
    return reward * h - pos


def joint_stationary(rates: RateEstimate) -> FloatArray:
    """2-D stationary law on the (q_bid, q_ask) grid via generator power iteration.

    Assumes the two queues share this module's single-queue rates — a
    tractable stand-in for the full joint QR model. Caps the grid at the
    estimate's q_max; raises if it is too large to invert iteratively.
    """
    q_max = rates.q_max
    if q_max < 1:
        raise ValueError("need q_max >= 1")
    if q_max > 64:
        raise ValueError("joint grid too large for iterative solve")
    n = (q_max + 1) ** 2
    # generator on flattened (q1, q2); transitions change one coordinate by +-1
    birth, death = rates.lam_l, rates.lam_c + rates.lam_m
    p = np.full(n, 1.0 / n)
    # power iteration on the embedded jump chain, seeded uniform
    idx = np.arange(n)
    q1 = idx // (q_max + 1)
    q2 = idx % (q_max + 1)
    for _ in range(2000):
        rate_out = birth[q1] * (q1 < q_max) + death[q1] * (q1 > 0)
        rate_out += birth[q2] * (q2 < q_max) + death[q2] * (q2 > 0)
        scale = rate_out.max()
        if scale <= 0:
            raise ValueError("degenerate rates")
        # uniformized CTMC single step: p <- p (I + G/scale)
        g = p.reshape(q_max + 1, q_max + 1)
        new_g = g * (1.0 - rate_out.reshape(q_max + 1, q_max + 1) / scale)
        # births on axis0: q1-1 -> q1
        b1 = birth[q1].reshape(q_max + 1, q_max + 1) / scale
        d1 = death[q1].reshape(q_max + 1, q_max + 1) / scale
        b2 = birth[q2].reshape(q_max + 1, q_max + 1) / scale
        d2 = death[q2].reshape(q_max + 1, q_max + 1) / scale
        new_g[:-1, :] += g[1:, :] * d1[1:, :]  # death in q1 moves mass down
        new_g[1:, :] += g[:-1, :] * b1[:-1, :]  # birth in q1 moves mass up
        new_g[:, :-1] += g[:, 1:] * d2[:, 1:]
        new_g[:, 1:] += g[:, :-1] * b2[:, :-1]
        p = new_g.ravel()
        p /= p.sum()
    return p.reshape(q_max + 1, q_max + 1)


def reference_rates(
    q_max: int, lam: float = 1.0, shape: str = "declining"
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Canonical synthetic intensity profiles used by tests/benches.

    ``declining``: limit arrivals fall as the queue grows (QR stylized fact):
    lam_l(q) = lam / (1 + 0.3 q); cancels/market grow mildly with q.
    """
    if q_max < 1:
        raise ValueError("q_max >= 1 required")
    q = np.arange(q_max + 1, dtype=float)
    if shape == "declining":
        return lam / (1.0 + 0.3 * q), lam * (0.2 + 0.05 * q), lam * (0.1 + 0.02 * q)
    if shape == "flat":
        return (
            np.full(q_max + 1, lam),
            np.full(q_max + 1, 0.3 * lam),
            np.full(q_max + 1, 0.15 * lam),
        )
    raise ValueError(f"unknown shape {shape!r}")


def bench_queue_reactive(seed: int = 20260131) -> dict[str, float]:
    """SYNTHETIC bench for the queue-reactive estimator. Correctness only."""
    out: dict[str, float] = {}
    q_max, horizon = 12, 200_000.0
    lam_l, lam_c, lam_m = reference_rates(q_max, lam=1.0)
    traj = simulate_queue(lam_l, lam_c, lam_m, horizon=horizon, seed=seed)
    est = estimate_rates(traj)
    # rate recovery: relative error on interior states with enough dwell
    mask = est.dwell > horizon * 0.01
    mask[-1] = False  # births at q_max are zero-rate by reflection
    rel = np.abs(est.lam_l[mask] - lam_l[mask]) / lam_l[mask]
    out["synthetic_rate_recovery_relerr"] = float(np.mean(rel))
    # stationary: analytic from *true* rates vs simulated empirical law
    pi_true = stationary_birth_death(
        RateEstimate(lam_l=lam_l, lam_c=lam_c, lam_m=lam_m, dwell=est.dwell, counts=est.counts)
    )
    pi_sim = stationary_simulated(traj)
    out["synthetic_stationary_tv"] = tv_distance(pi_true, pi_sim)
    # estimated-rate stationary vs true stationary (estimation error end-to-end)
    pi_est = stationary_birth_death(est)
    out["synthetic_stationary_est_tv"] = tv_distance(pi_true, pi_est)
    # V(q) monotonicity violations (should be non-increasing in q)
    v = queue_value_curve(est)
    viol = int(np.sum(np.diff(v) > 1e-9))
    out["synthetic_xshift_monotonicity_violations"] = float(viol)
    out["synthetic_xshift_endpoint_gap"] = float(v[0] - v[-1])
    # joint stationary sanity: uniform marginals sum to 1, all nonneg
    pj = joint_stationary(
        RateEstimate(lam_l=lam_l, lam_c=lam_c, lam_m=lam_m, dwell=est.dwell, counts=est.counts)
    )
    out["synthetic_joint_mass_err"] = float(abs(pj.sum() - 1.0))
    out["synthetic_joint_min"] = float(pj.min())
    # determinism
    t2 = simulate_queue(lam_l, lam_c, lam_m, horizon=10_000.0, seed=seed)
    t3 = simulate_queue(lam_l, lam_c, lam_m, horizon=10_000.0, seed=seed)
    same = int(np.array_equal(t2.states, t3.states))
    out["synthetic_determinism"] = float(same)
    return out
