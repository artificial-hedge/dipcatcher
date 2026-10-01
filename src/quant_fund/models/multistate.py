"""Continuous-time Markov multi-state survival models.

Modules
-------
* ``ctmc_intensity_fit`` — maximum-likelihood estimation of the generator
  ``Q`` for a continuous-time Markov chain from observed transition counts
  and statewise exposure times: ``q_ij = n_ij / u_i`` (the standard
  actuarial/biostatistic estimator, e.g. Kalbfleisch–Lawless 1985).
* ``transition_probabilities`` — ``P(s, t) = expm(Q (t − s))`` via matrix
  exponential; gives state-occupancy probabilities at any horizon.
* ``mean_sojourn`` — expected holding time in state ``i`` is ``-1/q_ii``.
* ``illness_death_simulate`` — event-driven simulation of the
  irreversible illness-death chain ``0 → 1, 0 → 2, 1 → 2``.
* ``aalen_johansen`` — nonparametric product-integral estimator of the
  transition-probability matrix for the illness-death chain under
  independent right-censoring (Aalen–Johansen 1978; Andersen et al. 1993).

Honesty contract
----------------
* All benches are SYNTHETIC event streams; state labels are abstract
  (0/1/2), not clinical diagnoses. No survival-probability marketing
  claims; outputs are estimation-accuracy diagnostics.
* No Sharpe/Sortino/Calmar/P&L metrics.

Composition
-----------
* Complements ``models/survival.py`` (single-endpoint Cox/KM) — this
  module handles competing/multi-state transition structure.
* ``metrics/hazard``-style diagnostics stay in survival; here we emit
  generator-level parameters for pricing/risk engines.

References
----------
* Aalen, O.O., Johansen, S. (1978), "An Empirical Transition Matrix for
  Non-Homogeneous Markov Chains Based on Censored Observations",
  Scandinavian Journal of Statistics 5:141–150.
* Andersen, P.K., Borgan, Ø., Gill, R.D., Keiding, N. (1993),
  Statistical Models Based on Counting Processes, Springer.
* Kalbfleisch, J.D., Lawless, J.F. (1985), "The Analysis of Panel Data
  under a Markov Assumption", Journal of the American Statistical
  Association 80:863–871.
* Jackson, C.H. (2011), "Multi-State Models for Panel Data: The msm
  Package for R", Journal of Statistical Software 38(8).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np
from scipy import linalg

FloatArray = np.ndarray


def ctmc_intensity_fit(
    transitions: Sequence[tuple[int, int]],
    exposure: FloatArray | dict[int, float],
    n_states: int = 3,
) -> FloatArray:
    """ML generator estimate: q_ij = n_ij / u_i (i ≠ j), row sums 0.

    Parameters
    ----------
    transitions
        List of ``(from_state, to_state)`` observed jumps.
    exposure
        Total observed time at risk per state — array of length
        ``n_states`` or dict keyed by state.
    n_states
        Chain size.

    Raises
    ------
    ValueError
        On unknown states or zero exposure for a state with exits.
    """
    if n_states < 2:
        raise ValueError("n_states >= 2")
    u = np.zeros(n_states)
    if isinstance(exposure, dict):
        for k, v in exposure.items():
            u[int(k)] = float(v)
    else:
        arr = np.asarray(exposure, dtype=np.float64).ravel()
        if arr.size != n_states:
            raise ValueError("exposure length must equal n_states")
        u = arr
    n_ij = np.zeros((n_states, n_states))
    for a, b in transitions:
        if not (0 <= a < n_states and 0 <= b < n_states):
            raise ValueError("transition state out of range")
        if a == b:
            raise ValueError("self-transitions are not jumps")
        n_ij[a, b] += 1.0
    q = np.zeros((n_states, n_states))
    for i in range(n_states):
        if n_ij[i].sum() > 0 and u[i] <= 0:
            raise ValueError(f"state {i} has exits but zero exposure")
        if u[i] > 0:
            q[i] = n_ij[i] / u[i]
    for i in range(n_states):
        q[i, i] = -q[i].sum()
    return q


def transition_probabilities(q: FloatArray, horizon: float) -> FloatArray:
    """P(0→h) = expm(Q·h). Validates generator structure."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be square")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if np.any(np.diag(q) > 1e-9) or np.any(np.abs(q.sum(axis=1)) > 1e-6):
        raise ValueError("q is not a valid CTMC generator")
    return np.asarray(linalg.expm(q * horizon))


def mean_sojourn(q: FloatArray) -> FloatArray:
    """Expected holding time per state: -1/q_ii (inf for absorbing)."""
    q = np.asarray(q, dtype=np.float64)
    d = np.diag(q)
    with np.errstate(divide="ignore"):
        out = np.where(d < -1e-12, -1.0 / d, np.inf)
    return np.asarray(out)


def illness_death_simulate(
    q01: float,
    q02: float,
    q12: float,
    n: int = 500,
    horizon: float = 20.0,
    censor_rate: float = 0.0,
    seed: int = 0,
) -> list[tuple[float, int, float, float]]:
    """Simulate the irreversible illness-death chain 0→1, 0→2, 1→2.

    Returns one tuple ``(t_leave0, dest0, t_death, t_censor)`` per
    subject:

    * ``t_leave0`` — time the subject leaves state 0 (to illness or
      death); equals ``min(t_censor, jump_time)``.
    * ``dest0`` — state entered at ``t_leave0``: 1 or 2, or ``-1`` when
      the observation was censored in state 0.
    * ``t_death`` — time of the 1→2 jump when the subject was observed
      ill first and died in view; ``inf`` otherwise.
    * ``t_censor`` — the administrative/random censoring time.
    """
    if min(q01, q02) <= 0 or q12 < 0 or horizon <= 0:
        raise ValueError("positive intensities/horizon required")
    rng = np.random.default_rng(seed)
    out: list[tuple[float, int, float, float]] = []
    for _ in range(n):
        t_cen = (
            horizon if censor_rate <= 0 else min(horizon, float(rng.exponential(1.0 / censor_rate)))
        )
        t0 = float(rng.exponential(1.0 / (q01 + q02)))
        if t0 >= t_cen:
            out.append((t_cen, -1, math.inf, t_cen))
            continue
        dest = 1 if rng.random() < q01 / (q01 + q02) else 2
        if dest == 2 or dest == 1:
            if dest == 1:
                t_d = t0 + float(rng.exponential(1.0 / q12)) if q12 > 0 else math.inf
                if t_d <= t_cen:
                    out.append((t0, 1, t_d, t_cen))
                else:
                    out.append((t0, 1, math.inf, t_cen))
            else:
                out.append((t0, 2, t0, t_cen))
    return out


def aalen_johansen_id(
    subjects: Sequence[tuple[float, int, float, float]],
    times: FloatArray,
) -> FloatArray:
    """Aalen–Johansen transition probabilities for illness-death.

    For each ``t`` in ``times`` returns the 3×3 matrix ``P(0, t)`` of
    state occupancy given start in state 0 — product-integrated over
    Nelson–Aalen increments. Censoring is assumed independent
    (state-independent).
    """
    grid = np.asarray(times, dtype=np.float64).ravel()
    if grid.size == 0 or np.any(grid <= 0):
        raise ValueError("times must be positive")
    rows = [tuple(map(float, subj)) for subj in subjects]
    if not rows:
        raise ValueError("no subjects")
    p_out = np.empty((grid.size, 3, 3))
    for gi, tt in enumerate(grid):
        p = np.eye(3)
        # event times up to tt, sorted
        events: list[tuple[float, int, int]] = []
        for t0, dest0, t_d, _tc in rows:
            if t0 <= tt and dest0 in (1, 2):
                events.append((t0, 0, int(dest0)))
            if math.isfinite(t_d) and t_d <= tt:
                events.append((t_d, 1, 2))
        events.sort()
        for t_ev, a, b_f in events:
            b = int(b_f)
            y0 = sum(1 for t0, d0, _td, tc in rows if t0 >= t_ev and tc >= t_ev)
            y1 = sum(
                1 for t0, d0, td, tc in rows if d0 == 1 and t0 < t_ev and td >= t_ev and tc >= t_ev
            )
            da = np.zeros((3, 3))
            if a == 0 and y0 > 0:
                incr = 1.0 / y0
                da[0, b] = incr
                da[0, 0] = -incr
            elif a == 1 and y1 > 0:
                incr = 1.0 / y1
                da[1, 2] = incr
                da[1, 1] = -incr
            p = p @ (np.eye(3) + da)
        p_out[gi] = p
    return p_out


def synth_multistate(seed: int = 0, n: int = 800) -> dict[str, object]:
    """SYNTHETIC illness-death with known intensities (q01=.15,q02=.05,q12=.3)."""
    rows = illness_death_simulate(0.15, 0.05, 0.30, n=n, horizon=25.0, censor_rate=0.02, seed=seed)
    exposure = np.zeros(3)
    transitions: list[tuple[int, int]] = []
    for t0, dest0, t_d, tc in rows:
        exposure[0] += t0
        if dest0 == 1:
            transitions.append((0, 1))
            exposure[1] += min(t_d, tc) - t0
            if math.isfinite(t_d):
                transitions.append((1, 2))
        elif dest0 == 2:
            transitions.append((0, 2))
    return {
        "rows": rows,
        "exposure": exposure,
        "transitions": transitions,
        "q_true": np.array([[-0.2, 0.15, 0.05], [0.0, -0.3, 0.3], [0.0, 0.0, 0.0]]),
    }


def bench_multistate(seed: int = 20261231 + 174) -> dict[str, float]:
    """SYNTHETIC ordering: CTMC ML + Aalen-Johansen recover truth."""
    d = synth_multistate(seed=seed)
    exposure = np.asarray(d["exposure"])
    transitions = cast(list[tuple[int, int]], d["transitions"])
    rows = cast(list[tuple[float, int, float, float]], d["rows"])
    q_hat = ctmc_intensity_fit(transitions, exposure, n_states=3)
    q_true = np.asarray(d["q_true"])
    rel = float(
        np.linalg.norm(q_hat[np.triu_indices(3, 1)] - q_true[np.triu_indices(3, 1)])
        / np.linalg.norm(q_true[np.triu_indices(3, 1)])
    )
    p_hat = transition_probabilities(q_hat, 5.0)
    p_true = transition_probabilities(q_true, 5.0)
    p_err = float(np.linalg.norm(p_hat - p_true) / np.linalg.norm(p_true))
    aj = aalen_johansen_id(rows, np.array([5.0]))
    aj_err = float(np.linalg.norm(aj[0] - p_true) / np.linalg.norm(p_true))
    soj = mean_sojourn(q_hat)
    soj_err = float(abs(soj[0] - 1.0 / 0.2) / (1.0 / 0.2))
    e1 = ctmc_intensity_fit(transitions, exposure, n_states=3)
    return {
        "synthetic_intensity_relerr": rel,
        "synthetic_pmat_relerr": p_err,
        "synthetic_aj_relerr": aj_err,
        "synthetic_sojourn0_err": soj_err,
        "synthetic_n_transitions": float(len(transitions)),
        "synthetic_determinism": float(np.allclose(e1, q_hat)),
    }
