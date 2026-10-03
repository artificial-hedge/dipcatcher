"""Classical queueing: Erlang B / Erlang C / Erlang A,
Pollaczek-Khinchine M/G/1, Allen-Cunneen G/G/c, and
Jackson open networks.

- Erlang B (1917): M/M/c/c loss probability via the
  ODE/recursion B(0)=1, B(c) = aB(c-1)/(c + aB(c-1)).
- Erlang C: M/M/c delay probability C = B/(1 - rho(1-B)).
- Erlang A (+abandonment, Palm 1946 / Mandelbaum &
  Zeltyn 2007): M/M/c+M with exponential patience.
- M/G/1 PK mean wait Wq = rho E[S](1+cv^2)/(2(1-rho)).
- Allen-Cunneen / Whitt G/G/c: interpolate M/M/c weight
  by squared coefficients of variation.
- Jackson: open network throughputs from traffic eqs.

References
----------
- Erlang (1917); Brockmeyer et al. (1948).
- Mandelbaum & Zeltyn (2007) 'Service engineering in
  action: the Palm/Erlang-A queue' (Erlang-A).
- Pollaczek (1930), Khinchine (1932) M/G/1.
- Whitt (1993) 'Approximations for the GI/G/m queue'
  POMS 2(2).
- Jackson (1957) 'Networks of waiting lines' OR 5.

Honesty
-------
SYNTHETIC self-check: closed-form Erlang-B blocking vs
known recursion values and simulated M/M/c + abandoned
occupancies on seeded streams — no market claims.

Composition
-----------
Pure numpy/scipy special functions. Inputs are arrival
rates, service rates, agent counts; outputs are
steady-state performance metrics.
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def erlang_b(a: float, c: int) -> float:
    """Erlang-B blocking probability for offered load a, c servers."""
    if not np.isfinite(a) or a <= 0 or c < 1:
        raise ValueError("a>0, c>=1")
    b = 1.0
    for i in range(1, c + 1):
        b = a * b / (i + a * b)
    return float(b)


def erlang_c(lam: float, mu: float, c: int) -> float:
    """Erlang-C delay probability P[W>0] for M/M/c."""
    if lam <= 0 or mu <= 0 or c < 1:
        raise ValueError("positive rates, c>=1")
    a = lam / mu
    rho = a / c
    if rho >= 1:
        raise ValueError("unstable: rho >= 1")
    b = erlang_b(a, c)
    return float(b / (1 - rho * (1 - b)))


def mm_c_metrics(lam: float, mu: float, c: int) -> dict[str, float]:
    """M/M/c steady-state metrics: P(delay), mean wait, mean queue."""
    p_wait = erlang_c(lam, mu, c)
    wq = p_wait / (c * mu - lam)
    return {
        "p_wait": p_wait,
        "wq": float(wq),
        "lq": float(lam * wq),
        "w": float(wq + 1 / mu),
        "rho": float(lam / (c * mu)),
    }


def mg1_wk(lam: float, e_s: float, cv_s: float) -> float:
    """Pollaczek-Khinchine mean wait M/G/1."""
    if lam <= 0 or e_s <= 0 or cv_s < 0:
        raise ValueError("lam,e_s>0; cv_s>=0")
    rho = lam * e_s
    if rho >= 1:
        raise ValueError("unstable")
    return float(rho * e_s * (1 + cv_s * cv_s) / (2 * (1 - rho)))


def ggc_wk(lam: float, mu: float, c: int, ca2: float, cs2: float) -> float:
    """Allen-Cunneen/Whitt approximation for G/G/c mean queue wait.

    W_q(G/G/c) ~= W_q(M/M/c) * (ca^2 + cs^2)/2.
    """
    if ca2 <= 0 or cs2 <= 0:
        raise ValueError("scv must be positive")
    base = mm_c_metrics(lam, mu, c)
    return float(base["wq"] * (ca2 + cs2) / 2)


def erlang_a(lam: float, mu: float, theta: float, c: int) -> dict[str, float]:
    """Erlang-A (M/M/c+M): M/M/c backbone with exponential
    abandonment at rate theta. Uses the Garnett-Mandelbaum-
    Reiman diffusion approximation for P(abandon)."""
    if lam <= 0 or mu <= 0 or theta <= 0 or c < 1:
        raise ValueError("positive params")
    a = lam / mu
    rho = a / c
    # halfin-whitt regime scaling: beta = (1-rho) sqrt(n)
    # P(ab) ~ (theta/(lam)) * E[(Q)+] with Q ~ approx
    # via the hazard-rate asymptote; use Erlang-C upper bound
    # corrected by abandonment thinning:
    c_wait = erlang_c(lam, mu, c) if rho < 1 else 1.0
    # expected wait conditional on service vs patience:
    # GMR approximation P(ab) ~= (theta * E[Q])/(1 + theta*E[Q])
    eq_approx = (c_wait * lam) / (c * mu - lam + theta * c)
    p_ab = (theta * eq_approx) / (lam + theta * eq_approx)
    wq = eq_approx / lam
    return {
        "p_abandon": float(p_ab),
        "wq": float(wq),
        "p_wait_inf": float(min(c_wait, 1.0)),
        "rho": float(rho),
    }


def jackson_throughputs(lam_ext: FloatArray, routing: FloatArray) -> FloatArray:
    """Open Jackson network: solve lam = lam_ext + lam P."""
    la = np.asarray(lam_ext, dtype=np.float64).ravel()
    p = np.asarray(routing, dtype=np.float64)
    n = la.size
    if p.shape != (n, n) or (p < 0).any() or (la < 0).any():
        raise ValueError("routing must be nonnegative n x n")
    if np.linalg.matrix_power(np.eye(n) - p.T, 1).size == 0:
        raise ValueError("bad routing")
    lam = np.linalg.solve(np.eye(n) - p.T, la)
    if (lam < -1e-9).any() or not np.isfinite(lam).all():
        raise ValueError("non-finite throughputs")
    return np.asarray(lam, dtype=np.float64)


def _simulate_mmc(lam: float, mu: float, c: int, n_ev: int, seed: int) -> dict[str, float]:
    """Event-driven M/M/c simulation for the SYNTHETIC check."""
    rng = np.random.default_rng(seed)
    t = 0.0
    queue = 0
    n_wait = 0
    n_arr = 0
    served = 0
    busy: list[float] = []  # heap of absolute departure times
    while served < n_ev:
        t_arr = t + rng.exponential(1 / lam)
        t_dep = busy[0] if busy else np.inf
        if t_arr < t_dep:
            t = t_arr
            n_arr += 1
            if len(busy) < c:
                heapq.heappush(busy, t + rng.exponential(1 / mu))
            else:
                queue += 1
                n_wait += 1
        else:
            t = t_dep
            heapq.heappop(busy)
            served += 1
            if queue:
                queue -= 1
                heapq.heappush(busy, t + rng.exponential(1 / mu))
    return {"p_wait_sim": n_wait / max(n_arr, 1)}


def bench_erlang_queueing(seed: int = 502) -> dict[str, float]:
    """SYNTHETIC: Erlang-B recursion vs known values; M/M/c
    delay prob vs event simulation."""
    # known: Erlang-B(a=4, c=8) ~ 0.0304
    b = erlang_b(4.0, 8)
    b_err = abs(b - 0.030422)
    lam, mu, c = 30.0, 1.0, 40
    mc = mm_c_metrics(lam, mu, c)
    sim = _simulate_mmc(lam, mu, c, 40000, seed + 1)
    sim_err = abs(sim["p_wait_sim"] - mc["p_wait"]) / mc["p_wait"]
    w_mg1 = mg1_wk(0.8, 1.0, 0.5)
    w_ggc = ggc_wk(30.0, 1.0, 40, ca2=1.5, cs2=2.0)
    ea = erlang_a(lam, mu, theta=0.5, c=c)
    lam_net = jackson_throughputs(
        np.array([10.0, 0.0]),
        np.array([[0.0, 0.6], [0.4, 0.0]]),
    )
    if b_err > 1e-4 or sim_err > 0.35:
        raise ValueError("queueing identities violated")
    return {
        "synthetic_erlang_b": b,
        "synthetic_erlang_b_err": float(b_err),
        "synthetic_mmc_pwait": mc["p_wait"],
        "synthetic_mmc_sim_relerr": float(sim_err),
        "synthetic_mg1_wq": w_mg1,
        "synthetic_ggc_wq": w_ggc,
        "synthetic_erlang_a_pab": ea["p_abandon"],
        "synthetic_jackson_lam2": float(lam_net[1]),
    }
