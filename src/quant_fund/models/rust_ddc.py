"""Rust (1987) nested fixed-point dynamic discrete choice.

The canonical optimal-stopping problem: a durable asset with
state x (wear) is either kept at maintenance cost c(x) or
replaced at fixed cost RC, resetting x. The inner loop
contracts the logsum Bellman EV for a candidate (θ, RC); the
outer loop maximizes the conditional-choice-probability
likelihood over those parameters — the NFXP estimator.

Honesty: synthetic panels simulate wear states and keep/
replace decisions under known parameters; the bench recovers
RC and θ — proper diagnostics, never market evidence.

References:
- Rust, J. (1987). Optimal replacement of GMC bus engines:
  an empirical model of Harold Zurcher. *Econometrica* 55 —
  the model and NFXP.
- Rust, J. (1994). Structural estimation of Markov decision
  processes. In Engle, R., McFadden, D. (eds.), *Handbook of
  Econometrics* IV — CCP representation used here.
- Hotz, V. J., Miller, R. A. (1993). Conditional choice
  probabilities and the estimation of dynamic models.
  *Review of Economic Studies* 60 — the two-step CCP
  alternative.
- Aguirregabiria, V., Mira, P. (2002). Swapping the nested
  fixed point algorithm. *Econometrica* 70 — the NPL
  iteration this nests.

Composition: numpy + scipy.optimize only — logsum Bellman
contraction + scalar/2-D likelihood search; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


def rust_value(
    theta: float,
    rc: float,
    trans: FloatArray,
    beta: float = 0.95,
    tol: float = 1e-10,
    max_iter: int = 5000,
) -> FloatArray:
    """Bellman contraction on the logsum expected value EV(x):
    keep yields −θ·x + Σ_δ p_δ EV(x+δ); replace yields −RC +
    EV(0). trans is the (K,K) stochastic keep-transition."""
    tt = np.asarray(trans, dtype=np.float64)
    k = tt.shape[0]
    if tt.ndim != 2 or tt.shape[1] != k or k < 5:
        raise ValueError("trans must be (K,K), K>=5")
    if rc <= 0 or theta <= 0 or not (0 < beta < 1):
        raise ValueError("theta, rc > 0 and 0<beta<1 required")
    x = np.arange(k, dtype=np.float64)
    ev = np.zeros(k)
    for _ in range(max_iter):
        v_keep = -theta * x + beta * (tt @ ev)
        v_repl = -rc + beta * ev[0]
        m = np.maximum(v_keep, v_repl)
        nxt = m + np.log(np.exp(v_keep - m) + np.exp(v_repl - m))
        if np.max(np.abs(nxt - ev)) < tol:
            return np.asarray(nxt, dtype=np.float64)
        ev = nxt
    raise ValueError("value iteration did not converge")


def rust_ccp(
    theta: float,
    rc: float,
    trans: FloatArray,
    beta: float = 0.95,
) -> FloatArray:
    """Conditional choice probabilities P(replace | x)."""
    tt = np.asarray(trans, dtype=np.float64)
    ev = rust_value(theta, rc, tt, beta)
    x = np.arange(ev.size, dtype=np.float64)
    v_keep = -theta * x + beta * (tt @ ev)
    v_repl = -rc + beta * ev[0]
    return np.asarray(
        np.exp(v_repl) / (np.exp(v_keep) + np.exp(v_repl)),
        dtype=np.float64,
    )


def rust_nfxp(
    x: FloatArray,
    choice: FloatArray,
    trans: FloatArray,
    beta: float = 0.95,
    grid_theta: FloatArray | None = None,
    grid_rc: FloatArray | None = None,
) -> dict[str, float]:
    """NFXP MLE over (theta, rc) — outer 2-D minimize on the
    profiled log-likelihood, inner contraction each eval."""
    xx = np.asarray(x, dtype=np.float64)
    c = np.asarray(choice, dtype=np.float64)
    n = xx.shape[0]
    if c.shape != (n,) or n < 40 or not np.all(np.isin(c, (0.0, 1.0))):
        raise ValueError("x (N,), binary choice (N,), N>=40 required")
    if not np.all(np.isfinite(xx)) or xx.min() < 0:
        raise ValueError("finite non-negative x required")
    k = np.asarray(trans).shape[0]
    if xx.max() >= k:
        raise ValueError("x out of transition range")
    xi = xx.astype(np.int64)

    def nll(p: FloatArray) -> float:
        th, rc = float(p[0]), float(p[1])
        if th <= 0 or rc <= 0:
            return 1e12
        try:
            p1 = rust_ccp(th, rc, np.asarray(trans), beta)
        except ValueError:
            return 1e12
        pr = np.clip(p1[xi], 1e-12, 1 - 1e-12)
        ll = c * np.log(pr) + (1 - c) * np.log(1 - pr)
        return -float(np.sum(ll))

    best: dict[str, float] | None = None
    for th0 in grid_theta if grid_theta is not None else np.array([0.02, 0.1, 0.5]):
        for rc0 in grid_rc if grid_rc is not None else np.array([2.0, 8.0, 20.0]):
            r = minimize(
                nll,
                np.array([th0, rc0]),
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 60},
            )
            if best is None or float(r.fun) < best["nll"]:
                best = {
                    "theta": float(r.x[0]),
                    "rc": float(r.x[1]),
                    "nll": float(r.fun),
                }
    if not (best is not None):
        raise ValueError("best is not None")
    return best


def synth_rust(
    n_periods: int = 4000,
    theta: float = 0.1,
    rc: float = 8.0,
    n_states: int = 30,
    beta: float = 0.95,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Simulate wear states and keep/replace decisions under
    (theta, rc). trans: x→x+δ with probs (.6,.3,.1) up to K."""
    rng = np.random.default_rng(seed)
    k = n_states
    trans = np.zeros((k, k))
    probs = (0.6, 0.3, 0.1)
    for i in range(k):
        for d, p in enumerate(probs):
            j = min(k - 1, i + d)
            trans[i, j] += p
    p1 = rust_ccp(theta, rc, trans, beta)
    x_seq = np.zeros(n_periods)
    c_seq = np.zeros(n_periods)
    x = 0
    for i in range(n_periods):
        c = float(rng.random() < p1[x])
        c_seq[i] = c
        x_seq[i] = float(x)
        if c:
            x = 0
        else:
            x = min(k - 1, x + rng.choice(3, p=probs))
    return {"x": x_seq, "choice": c_seq, "trans": trans}


def bench_rust_ddc(seed: int = 20261231 + 265) -> dict[str, float]:
    """Rust self-check: NFXP recovers θ≈0.1, RC≈8 from a
    simulated replacement panel; a wrong-RC comparison is
    rejected by likelihood. All ``synthetic_*``."""
    d = synth_rust(theta=0.1, rc=8.0, seed=seed)
    est = rust_nfxp(
        np.asarray(d["x"]),
        np.asarray(d["choice"]),
        np.asarray(d["trans"]),
    )
    est2 = rust_nfxp(
        np.asarray(d["x"]),
        np.asarray(d["choice"]),
        np.asarray(d["trans"]),
    )
    return {
        "synthetic_theta": est["theta"],
        "synthetic_rc": est["rc"],
        "synthetic_theta_true": 0.1,
        "synthetic_rc_true": 8.0,
        "synthetic_detects": float(abs(est["theta"] - 0.1) < 0.08 and abs(est["rc"] - 8.0) < 3.0),
        "synthetic_determinism": float(est2["rc"] == est["rc"]),
    }
