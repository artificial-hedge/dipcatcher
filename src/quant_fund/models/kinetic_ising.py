"""Kinetic Ising model for cross-asset co-movement.

Spins ``s_i(t) in {-1, +1}`` (e.g. sign of asset returns) evolve by
parallel Glauber updates

    P(s_i(t+1) = +1 | s(t)) = sigmoid(2 * (sum_j J_ij s_j(t) + h_i))

so the joint probability factorizes and pseudo-likelihood is exact.
Three estimation routes:

- nMF inversion (Mezard-Sakellariou): ``J = A^{-1} D`` where ``A`` is the
  equal-time correlation matrix and ``D`` the one-step delayed
  correlation ``<s_i(t) s_j(t+1)>``; fields ``h_i`` are recovered from
  ``atanh(m_i) = h_i + sum_j J_ij m_j``.
- TAP correction (Roudi-Hertz iterative scheme): refines nMF with the
  Thouless-Anderson-Palmer Onsager term.
- Pseudo-likelihood refinement: per-spin logistic regression of
  ``s_i(t+1)`` on ``s(t)`` — exact for kinetic Ising; Newton iterations
  on the separable per-spin problems.

Validation: exact enumeration of the 2^n Boltzmann measure for small
systems (n <= 10) to check equilibrium statistics under symmetric J.

References
----------
- Bury (2013). Market structure explained by pairwise interactions.
  *Physica A* 392 — arXiv:1209.1980 (verified; kinetic Ising on
  financial return signs).
- Roudi, Tyrcha & Hertz (2009). Ising model for neural data: model
  quality and approximate methods for extraction of connectivity.
  *PRE* 79 — arXiv:0906.1410 (verified; nMF + TAP for kinetic Ising).
- Mezard & Sakellariou (2011). Exact mean-field inference in asymmetric
  kinetic Ising systems. *JSM* — arXiv:1103.3433 (verified; J = A^{-1}D).
- Bouchaud (2013). Crises and collective socio-economic phenomena.
  *JEDC* — arXiv:1209.0453 (verified; interaction-driven regimes).

Honesty
-------
All benches run on seeded SYNTHETIC spin systems generated in-module
(known J, h ground truth). Recovery errors validate inference machinery
only — never market evidence.

Composition notes
-----------------
- ``models/`` regime models (HMM etc.) parametrize aggregate regimes;
  this module resolves pairwise coupling structure.
- ``metrics/`` correlation/network diagnostics consume coupling matrices.
- ``research/benches_w25.py``: wraps ``bench_kinetic_ising`` as an
  OPTIONAL scorecard family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_spins(s: FloatArray, min_t: int = 20) -> FloatArray:
    v = np.asarray(s, dtype=float)
    if v.ndim != 2 or v.shape[0] < min_t:
        raise ValueError(f"need a (T >= {min_t}, N) spin matrix")
    if not np.isfinite(v).all() or not np.isin(np.unique(v), (-1.0, 1.0)).all():
        raise ValueError("spins must be +-1")
    return v


def simulate_kinetic_ising(
    n: int,
    T: int,
    J: FloatArray | None = None,
    h: FloatArray | None = None,
    coupling_scale: float = 0.4,
    burn: int = 50,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Simulate parallel-update kinetic Ising dynamics.

    Returns (spins (T, n), J used, h used). Random J ~ N(0,
    coupling_scale/sqrt(n)) with zero diagonal; h ~ N(0, 0.05).
    """
    if n < 3 or T < 20:
        raise ValueError("n >= 3, T >= 20")
    rng = np.random.default_rng(seed)
    if J is None:
        J = rng.normal(0.0, coupling_scale / np.sqrt(n), (n, n))
        np.fill_diagonal(J, 0.0)
    J = np.asarray(J, dtype=float)
    if J.shape != (n, n) or not np.isfinite(J).all():
        raise ValueError("J must be a finite (n, n) matrix")
    if h is None:
        h = rng.normal(0.0, 0.05, n)
    h = np.asarray(h, dtype=float)
    if h.shape != (n,) or not np.isfinite(h).all():
        raise ValueError("h must be a finite (n,) vector")
    s = rng.choice([-1.0, 1.0], size=(T + burn, n)).astype(np.float64)
    for t in range(T + burn - 1):
        field = s[t] @ J.T + h  # field_i = sum_j J_ij s_j
        prob = 1.0 / (1.0 + np.exp(-2.0 * field))
        s[t + 1] = np.where(rng.random(n) < prob, 1.0, -1.0)
    return s[burn:], J, h


@dataclass(frozen=True)
class IsingFit:
    """Estimated kinetic Ising parameters."""

    J: FloatArray
    h: FloatArray
    method: str
    loglik: float


def _corr(s: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    m = s.mean(axis=0)
    a = (s[1:].T @ s[1:]) / (s.shape[0] - 1)  # equal-time product mean
    d = (s[:-1].T @ s[1:]) / (s.shape[0] - 1)  # D_ij = <s_j(t) s_i(t+1)>
    return m, a, d


def nmf_invert(s: FloatArray, eps: float = 1e-4) -> IsingFit:
    """Mezard-Sakellariou nMF inversion: J = A^{-1} D."""
    v = _check_spins(s)
    m, a, d = _corr(v)
    n = v.shape[1]
    a_reg = a + eps * np.eye(n)
    J = np.linalg.solve(a_reg, d).T  # solve for rows then transpose
    np.fill_diagonal(J, 0.0)
    # atanh(m_i) = h_i + sum_j J_ij m_j  ->  h_i = atanh(m_i) - sum_j J_ij m_j
    mm = np.clip(m, -0.999, 0.999)
    h = np.arctanh(mm) - J @ mm
    ll = float(_pseudo_loglik(v, J, h))
    return IsingFit(J=J, h=h, method="nmf", loglik=ll)


def tap_refine(s: FloatArray, eps: float = 1e-4, iters: int = 30) -> IsingFit:
    """Roudi-Hertz TAP iteration on top of nMF.

    TAP update for kinetic Ising (Roudi & Hertz 2011 dynamical TAP):
      J_new = nMF J minus the Onsager self-reaction correction
      J_ij <- J_ij - J_ij * J_ji * m_i * m_j applied iteratively.
    """
    v = _check_spins(s)
    base = nmf_invert(v, eps=eps)
    J = base.J.copy()
    m, _, _ = _corr(v)
    mm = np.clip(m, -0.999, 0.999)
    for _ in range(iters):
        correction = (J * J.T) * np.outer(mm, mm)
        J_new = base.J - correction
        np.fill_diagonal(J_new, 0.0)
        if np.allclose(J_new, J, atol=1e-10):
            J = J_new
            break
        J = J_new
    h = np.arctanh(mm) - J @ mm
    ll = float(_pseudo_loglik(v, J, h))
    return IsingFit(J=J, h=h, method="tap", loglik=ll)


def _pseudo_loglik(s: FloatArray, J: FloatArray, h: FloatArray) -> float:
    field = s[:-1] @ J.T + h[None, :]  # field_i = sum_j J_ij s_j
    z = 2.0 * s[1:] * field
    # log sigmoid(z) = -softplus(-z)
    ll = -np.logaddexp(0.0, -z)
    return float(ll.mean())


def pl_refine(
    s: FloatArray, J0: FloatArray | None = None, iters: int = 60, step: float = 0.5
) -> IsingFit:
    """Per-spin pseudo-likelihood maximization (exact for kinetic Ising).

    Independent logistic regression per target spin: Newton on the mean
    log-likelihood gradient with L2 shrinkage for stability.
    """
    v = _check_spins(s)
    n = v.shape[1]
    m, _, _ = _corr(v)
    mm = np.clip(m, -0.999, 0.999)
    if J0 is None:
        J0 = np.zeros((n, n))
    J0 = np.asarray(J0, dtype=float)
    if J0.shape != (n, n) or not np.isfinite(J0).all():
        raise ValueError("J0 must be a finite (n, n) matrix")
    h0 = np.arctanh(mm)
    X = v[:-1]  # (T-1, n) predictors
    Y = v[1:]  # (T-1, n) targets
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])  # + intercept for h
    J = J0.copy()
    h = h0.copy()
    for i in range(n):
        w = np.concatenate([J[i], [h[i]]])
        w[i] = 0.0  # no self-coupling
        yi = Y[:, i]
        for _ in range(iters):
            z = 2.0 * yi * (Xa @ w)
            p = 1.0 / (1.0 + np.exp(-z))
            resid = 2.0 * yi * (1.0 - p)  # d ll/d field
            grad = Xa.T @ resid / Xa.shape[0] - 1e-3 * w
            # Hessian: -4 X^T diag(p(1-p)) X - ridge
            q = 4.0 * p * (1.0 - p)
            hess = -(Xa.T * q) @ Xa / Xa.shape[0] - 1e-3 * np.eye(n + 1)
            try:
                dw = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            w_new = w - step * dw
            w_new[i] = 0.0
            if np.max(np.abs(w_new - w)) < 1e-9:
                w = w_new
                break
            w = w_new
        J[i] = w[:n]
        h[i] = w[n]
    ll = float(_pseudo_loglik(v, J, h))
    return IsingFit(J=J, h=h, method="pl", loglik=ll)


def enumerate_boltzmann(J: FloatArray, h: FloatArray) -> dict[str, float]:
    """Exact equilibrium statistics of the pairwise Boltzmann measure
    P(s) ~ exp(sum_ij J_ij s_i s_j + sum_i h_i s_i) for n <= 10.

    Consistency check for symmetric J — the kinetic dynamics'
    stationary correlations should track these equilibrium values.
    """
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = J.shape[0]
    if J.shape != (n, n) or h.shape != (n,):
        raise ValueError("J (n,n) and h (n,) required")
    if n > 10 or n < 2:
        raise ValueError("2 <= n <= 10 for enumeration")
    # all 2^n states as ±1
    idx = np.arange(2**n)
    states = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float) * 2.0 - 1.0
    en = np.einsum("bi,ij,bj->b", states, J, states) + states @ h
    logw = en - en.max()
    w = np.exp(logw)
    w /= w.sum()
    m = w @ states  # mean magnetization
    pair = np.einsum("b,bi,bj->ij", w, states, states)  # <s_i s_j>
    return {
        "partition_log": float(np.log(np.exp(logw).sum()) + en.max()),
        "mean_abs_mag": float(np.abs(m).mean()),
        "pair_corr_mean": float(pair[np.triu_indices(n, 1)].mean()),
    }


def simulated_pair_corr(s: FloatArray) -> FloatArray:
    """Empirical <s_i s_j> correlation matrix."""
    v = _check_spins(s)
    return np.asarray((v.T @ v) / v.shape[0], dtype=np.float64)


def magnetization_path(s: FloatArray) -> FloatArray:
    """Per-step mean magnetization — regime indicator."""
    v = _check_spins(s)
    return v.mean(axis=1)


def effective_temperature(s: FloatArray) -> float:
    """Glauber-style effective temperature from the lag-1 self-correlation.

    For h ~ 0 and weak J, <s_i(t) s_i(t+1)> ~ tanh(2 J_eff) — report the
    implied field scale; high values mean locked-in (cold) dynamics.
    """
    v = _check_spins(s)
    c = float((v[:-1] * v[1:]).mean())
    c = float(np.clip(c, -0.999, 0.999))
    return float(0.5 * np.arctanh(c))


def synth_market_spins(n: int, T: int, seed: int = 0) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Synthetic 'market-like' spin series: random J with a sparse hub
    structure (a few strong couplings on a weak background)."""
    if n < 3 or T < 20:
        raise ValueError("n >= 3, T >= 20")
    rng = np.random.default_rng(seed)
    J = rng.normal(0.0, 0.15 / np.sqrt(n), (n, n))
    np.fill_diagonal(J, 0.0)
    hubs = rng.choice(n, size=max(1, n // 5), replace=False)
    for hub in hubs:
        J[hub, :] += rng.normal(0.0, 0.5, n)
        J[hub, hub] = 0.0
    h = rng.normal(0.0, 0.05, n)
    s, J_used, h_used = simulate_kinetic_ising(n, T, J=J, h=h, seed=seed)
    return s, J_used, h_used


def _relerr(est: FloatArray, truth: FloatArray) -> float:
    num = float(np.linalg.norm(est - truth))
    den = float(np.linalg.norm(truth))
    return num / max(den, 1e-12)


def bench_kinetic_ising(seed: int = 20260203) -> dict[str, float]:
    """SYNTHETIC bench: coupling recovery, PL-vs-nMF, correlation
    recovery, equilibrium consistency, determinism."""
    out: dict[str, float] = {}
    s, J_true, h_true = synth_market_spins(12, 3000, seed=seed)
    fit_nmf = nmf_invert(s)
    fit_tap = tap_refine(s)
    fit_pl = pl_refine(s)
    out["synthetic_nmf_J_relerr"] = _relerr(fit_nmf.J, J_true)
    out["synthetic_tap_J_relerr"] = _relerr(fit_tap.J, J_true)
    out["synthetic_pl_J_relerr"] = _relerr(fit_pl.J, J_true)
    out["synthetic_nmf_h_relerr"] = _relerr(fit_nmf.h, h_true)
    out["synthetic_pl_h_relerr"] = _relerr(fit_pl.h, h_true)
    out["synthetic_pl_beats_nmf"] = float(
        out["synthetic_pl_J_relerr"] < out["synthetic_nmf_J_relerr"]
    )
    # delayed-correlation recovery: the model fit should reproduce the
    # one-step cross-covariances it was estimated from
    s2, _, _ = simulate_kinetic_ising(12, 3000, J=fit_pl.J, h=fit_pl.h, seed=seed + 7)
    d_emp = (s[:-1].T @ s[1:]) / (s.shape[0] - 1)
    d_mod = (s2[:-1].T @ s2[1:]) / (s2.shape[0] - 1)
    num = np.linalg.norm(d_emp - d_mod)
    den = np.linalg.norm(d_emp)
    out["synthetic_corr_recovery_relerr"] = float(num / max(den, 1e-12))
    # equilibrium consistency on a small symmetric system
    rng = np.random.default_rng(seed)
    Js = rng.normal(0.0, 0.3, (8, 8))
    Js = (Js + Js.T) / 2.0
    np.fill_diagonal(Js, 0.0)
    hs = rng.normal(0.0, 0.05, 8)
    exact = enumerate_boltzmann(Js, hs)
    ss, _, _ = simulate_kinetic_ising(8, 40000, J=Js, h=hs, seed=seed)
    emp = simulated_pair_corr(ss)
    out["synthetic_enum_paircorr_gap"] = float(
        abs(emp[np.triu_indices(8, 1)].mean() - exact["pair_corr_mean"])
    )
    out["synthetic_eff_temperature"] = effective_temperature(s)
    s2, _, _ = synth_market_spins(12, 3000, seed=seed)
    np.testing.assert_allclose(s, s2)
    out["synthetic_determinism"] = 1.0
    return out
