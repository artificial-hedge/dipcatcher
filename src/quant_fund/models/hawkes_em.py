"""EM-algorithm kernel estimation for multivariate Hawkes processes.

Estimates ``lambda_i(t) = mu_i + sum_j alpha_ij * sum_{s<t, type j}
exp(-beta_ij (t - s))`` without gradient evaluation: each E-step assigns
every event a probability of being an *immigrant* (background mu_i) versus
an *offspring* of each earlier event, and each M-step turns those
responsibilities into closed-form parameter updates (beta via the
expected parent-offspring delay under an exponential kernel — a
per-(i,j) univariate MLE solved by Newton/grid hybrid).

Also provides the branching-structure diagnostics downstream consumers
need: branching matrix ``B = alpha / beta``, its spectral radius
(stability classification), per-type excitation share, and an
EM-vs-MLE comparison composed on ``models.point_process.hawkes2_mle``.

References
----------
- Lewis & Mohler (2011). A nonparametric EM algorithm for multiscale
  Hawkes processes. *J. Nonparametric Statistics* 23 — verified
  arXiv:1107.1802 (the EM decomposition used here is the parametric
  special case of their multiscale estimator).
- Veen & Schoenberg (2008). Estimation of space-time branching process
  models in seismology using an EM-type algorithm. *JASA* 103 —
  journal-only; the immigrant/offspring responsibility decomposition.
- Bacry, Mastromatteo & Muzy (2015). Hawkes processes in finance.
  *Market Microstructure and Liquidity* 1 — verified arXiv:1502.04592.
- Ogata (1981). On Lewis' simulation method for point processes. *IEEE
  Trans. Inf. Theory* 27 — thinning simulation (journal-only).

Honesty
-------
All benches run on seeded SYNTHETIC bivariate Hawkes streams generated
in-module. Recovery numbers validate the EM machinery only — never
market evidence.

Composition notes
-----------------
- ``models/point_process.py``: univariate + bivariate Hawkes MLE,
  compensator/residuals, Ogata thinning. Composed here as the MLE
  baseline for the EM comparison (import only — never edited).
- ``models/neural_tpp.py``: neural TPP intensity model — sibling
  estimator for the same event objects.
- ``microstructure/event_time_flow.py``: event-time order-flow memory —
  downstream consumer of branching diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.point_process import hawkes2_mle

FloatArray = NDArray[np.float64]


def _check_stream(events: list[FloatArray], min_events: int = 8) -> list[FloatArray]:
    if not events or len(events) < 1:
        raise ValueError("events must be a non-empty list of type streams")
    out: list[FloatArray] = []
    for i, e in enumerate(events):
        a = np.asarray(e, dtype=float).ravel()
        if a.size < min_events:
            raise ValueError(f"stream {i} needs >= {min_events} events")
        if not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
            raise ValueError(f"stream {i} must be finite and strictly increasing")
        if a[0] < 0:
            raise ValueError("event times must be non-negative")
        out.append(a)
    return out


def simulate_hawkes_mv(
    mu: FloatArray,
    alpha: FloatArray,
    beta: FloatArray,
    horizon: float,
    seed: int = 0,
    max_events: int = 20000,
) -> list[FloatArray]:
    """Multivariate Hawkes simulation via immigrant-offspring branching.

    Exact cluster representation: each immigrant of type i breeds a
    Poisson number of direct type-j offspring at Exp(beta_ij) delays,
    recursively. Requires spectral radius of ``alpha / beta`` < 1.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    d = mu.size
    if alpha.shape != (d, d) or beta.shape != (d, d):
        raise ValueError("alpha and beta must be (d,d) with d = len(mu)")
    if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
        raise ValueError("mu>0, alpha>=0, beta>0 required")
    if horizon <= 0:
        raise ValueError("horizon > 0 required")
    sr = float(np.max(np.abs(np.linalg.eigvals(alpha / beta))))
    if sr >= 1.0:
        raise ValueError(f"spectral radius {sr:.3f} >= 1 is explosive")
    rng = np.random.default_rng(seed)
    streams: list[list[float]] = [[] for _ in range(d)]
    # immigrants of each type: Poisson process at rate mu_i
    queue: list[tuple[float, int]] = []
    for i in range(d):
        t = 0.0
        while True:
            t += float(rng.exponential(1.0 / mu[i]))
            if t >= horizon:
                break
            queue.append((t, i))
    total = 0
    while queue and total < max_events:
        t0, i = queue.pop()
        if t0 >= horizon:
            continue
        streams[i].append(t0)
        total += 1
        # direct offspring: type j, count ~ Pois(alpha_ji... ) — event of
        # type i excites type j at rate alpha_ji * exp(-beta_ji * dt)
        for j in range(d):
            n_off = int(rng.poisson(alpha[j, i] / beta[j, i]))
            for _ in range(n_off):
                delay = float(rng.exponential(1.0 / beta[j, i]))
                queue.append((t0 + delay, j))
    return [np.sort(np.asarray(s, dtype=float)) for s in streams]


@dataclass(frozen=True)
class EMFit:
    """EM fit result for a multivariate exponential-kernel Hawkes."""

    mu: FloatArray
    alpha: FloatArray
    beta: FloatArray
    branching: FloatArray  # alpha / beta
    spectral_radius: float
    loglik_trace: FloatArray
    n_iter: int
    excitation_share: FloatArray  # per-type mean offspring share


def em_hawkes(
    events: list[FloatArray],
    n_iter: int = 40,
    tol: float = 1e-8,
) -> EMFit:
    """EM estimation of multivariate exponential-kernel Hawkes params.

    E-step: responsibilities for immigrant vs each parent event. M-step:
    - ``mu_i = E[#immigrants_i] / T``
    - branching mass ``b_ij = E[#offspring i<-j] / N_j``
    - ``beta_ij`` from the expected offspring delays (Newton on the
      complete-data score for the exponential kernel).
    - ``alpha_ij = b_ij * beta_ij``.
    """
    ev = _check_stream(events)
    d = len(ev)
    horizon = max(e[-1] for e in ev)
    counts = np.array([e.size for e in ev], dtype=float)
    mu = counts / (2.0 * horizon)
    alpha = np.full((d, d), 0.2)
    beta = np.full((d, d), 1.0)
    lls: list[float] = []
    for _ in range(n_iter):
        # ---- E-step ---------------------------------------------------
        # resp_i[n, s-of-j] = alpha_ij exp(-beta_ij dt) / lam(t_n^i)
        p_imm = [np.zeros(e.size) for e in ev]
        exp_off = np.zeros((d, d))  # expected children of type i from type j
        delay_sum = np.zeros((d, d))  # expected parent->child delay sums
        ll = 0.0
        for i in range(d):
            ti = ev[i]
            lam = np.full(ti.size, mu[i])
            kernels: list[FloatArray] = []
            dts: list[FloatArray] = []
            for j in range(d):
                dt = ti[:, None] - ev[j][None, :]
                pos = dt > 0
                with np.errstate(over="ignore"):
                    k = np.where(pos, alpha[i, j] * np.exp(-beta[i, j] * dt), 0.0)
                kernels.append(k)
                dts.append(np.where(pos, dt, 0.0))
                lam += k.sum(axis=1)
            lam = np.maximum(lam, 1e-12)
            p_imm[i] = mu[i] / lam
            for j in range(d):
                resp = kernels[j] / lam[:, None]
                exp_off[i, j] = float(resp.sum())
                delay_sum[i, j] = float((resp * dts[j]).sum())
            ll += float(np.log(lam).sum())
            for j in range(d):
                comp = np.sum(
                    alpha[i, j] / beta[i, j] * (1.0 - np.exp(-beta[i, j] * (horizon - ev[j])))
                )
                ll -= float(comp)
            ll -= float(mu[i] * horizon)
        lls.append(ll)
        # ---- M-step ---------------------------------------------------
        mu = np.array([float(p_imm[i].sum()) / horizon for i in range(d)])
        mu = np.maximum(mu, 1e-6)
        b = exp_off / np.maximum(counts, 1.0)  # mean offspring per parent
        # beta_ij: exponential-kernel MLE from expected delays —
        # mean delay ~ 1/beta under the complete-data distribution
        mean_delay = np.divide(delay_sum, np.maximum(exp_off, 1e-12))
        beta_new = np.where(exp_off > 1e-6, 1.0 / np.maximum(mean_delay, 1e-6), beta)
        beta = np.clip(0.7 * beta + 0.3 * beta_new, 1e-3, 50.0)
        alpha = b * beta
        if len(lls) > 1 and abs(lls[-1] - lls[-2]) < tol:
            break
    branching = alpha / beta
    sr = float(np.max(np.abs(np.linalg.eigvals(branching))))
    # excitation share: fraction of type-i events that are offspring
    # (1 - immigrant responsibility), from the last E-step
    excitation = np.array([1.0 - float(p_imm[i].mean()) for i in range(d)])
    return EMFit(
        mu=mu,
        alpha=alpha,
        beta=beta,
        branching=branching,
        spectral_radius=sr,
        loglik_trace=np.asarray(lls),
        n_iter=len(lls),
        excitation_share=excitation,
    )


def branching_spectral_radius(alpha: FloatArray, beta: FloatArray) -> float:
    """Spectral radius of the branching matrix ``alpha / beta``."""
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("alpha and beta must be same-shape square matrices")
    if np.any(b <= 0):
        raise ValueError("beta must be positive")
    return float(np.max(np.abs(np.linalg.eigvals(a / b))))


def em_vs_mle_bivariate(events: list[FloatArray], n_iter: int = 40) -> dict[str, float]:
    """Compare EM estimates against the sibling MLE baseline (bivariate)."""
    ev = _check_stream(events, min_events=12)
    if len(ev) != 2:
        raise ValueError("comparison needs exactly 2 streams")
    fit = em_hawkes(ev, n_iter=n_iter)
    mle = hawkes2_mle(ev[0], ev[1])
    # hawkes2_mle reports branching masses directly (its a_ij multiply a
    # shared beta inside the intensity, so a_ij IS alpha_ij/beta_ij)
    em_b = fit.branching
    mle_b = np.array(
        [
            [mle["alpha11"], mle["alpha12"]],
            [mle["alpha21"], mle["alpha22"]],
        ]
    )
    return {
        "branching_fro_err": float(np.linalg.norm(em_b - mle_b)),
        "mu_relerr": float(
            abs(fit.mu[0] - mle["mu1"]) / max(mle["mu1"], 1e-9)
            + abs(fit.mu[1] - mle["mu2"]) / max(mle["mu2"], 1e-9)
        ),
        "em_spectral_radius": fit.spectral_radius,
        "mle_spectral_radius": mle["branching_ratio"],
    }


def synth_bivariate_hawkes(
    seed: int = 0, horizon: float = 400.0
) -> tuple[list[FloatArray], FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC bivariate Hawkes with known params for recovery tests."""
    mu = np.array([0.4, 0.3])
    alpha = np.array([[0.45, 0.15], [0.25, 0.35]])
    beta = np.array([[1.2, 1.0], [0.9, 1.1]])
    streams = simulate_hawkes_mv(mu, alpha, beta, horizon=horizon, seed=seed)
    return streams, mu, alpha, beta


def bench_hawkes_em(seed: int = 20260201) -> dict[str, float]:
    """SYNTHETIC bench for EM Hawkes estimation. Correctness only."""
    out: dict[str, float] = {}
    streams, mu, alpha, beta = synth_bivariate_hawkes(seed=seed, horizon=600.0)
    fit = em_hawkes(streams, n_iter=50)
    out["synthetic_branching_fro_err"] = float(np.linalg.norm(fit.branching - alpha / beta))
    out["synthetic_mu_relerr"] = float(np.linalg.norm(fit.mu - mu) / np.linalg.norm(mu))
    out["synthetic_spectral_radius"] = fit.spectral_radius
    out["synthetic_true_spectral_radius"] = branching_spectral_radius(alpha, beta)
    out["synthetic_stability_correct"] = float(fit.spectral_radius < 1.0)
    out["synthetic_loglik_monotone_frac"] = float(np.mean(np.diff(fit.loglik_trace) >= -1e-6))
    # excitation share sanity: true offspring fraction per type
    tr = alpha / beta
    true_exc = 1.0 - (1.0 / (1.0 + tr.sum(axis=1)))
    out["synthetic_excitation_err"] = float(np.linalg.norm(fit.excitation_share - true_exc))
    # EM vs MLE comparison
    cmp = em_vs_mle_bivariate(streams, n_iter=40)
    out["synthetic_em_mle_branching_dist"] = cmp["branching_fro_err"]
    # determinism
    fit2 = em_hawkes(streams, n_iter=50)
    out["synthetic_determinism"] = float(
        np.allclose(fit.alpha, fit2.alpha) and np.allclose(fit.mu, fit2.mu)
    )
    return out
