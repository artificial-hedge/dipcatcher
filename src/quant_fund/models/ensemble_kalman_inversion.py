"""Ensemble Kalman Inversion (EKI) — derivative-free parameter estimation (SYNTHETIC).

EKI treats inverse problems as sequential ensemble filtering: an ensemble
of parameters is iterated through Kalman-style updates driven by a black-box
forward map ``G(θ) → observation space``, with sample covariances supplying
the Jacobian-free sensitivity (Iglesias, Law & Stuart 2013). Unlike
likelihood-based Bayesian estimation it never evaluates a density — the
forward simulator is the whole model — which makes it the natural
complement to PMCMC for simulators whose likelihood is unavailable or
intractable (e.g. moment matching under a stochastic-volatility DGP).

Updates use the perturbed-observation (stochastic) EKI form
``θ_j⁺ = θ_j + C^{θg}(C^{gg} + Γ)⁻¹(y + η_j − G(θ_j))`` with
``η_j ~ N(0, Γ)``; ``tikhonov_eki`` augments the forward with a prior
regularization row-block. Optional multiplicative covariance inflation and
correlation localization (Gaspari–Cohn taper on the obs cross-covariance)
follow the ensemble-Kalman-filter literature.

Honesty
-------
All bench outputs are ``synthetic_*`` correctness diagnostics on seeded
synthetic inverse problems — parameter recovery error, misfit reduction,
ensemble spread collapse, SV-moment matching error. They verify estimator
mechanics, never market evidence. No Sharpe/Sortino/Calmar/P&L/NAV ever.

References
----------
- Iglesias, M.A., Law, K.J.H. & Stuart, A.M. (2013). Ensemble Kalman
  methods for inverse problems. *Inverse Problems* 29(4):045001.
- Schillings, C. & Stuart, A.M. (2017). Analysis of the ensemble Kalman
  filter for inverse problems. *SIAM Journal on Numerical Analysis*
  55(3):1264–1290. (Continuous-time limit and collapse theory.)
- Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman Filter*, 2nd
  ed. Springer. (Perturbed-observation EnKF and inflation.)
- Gaspari, G. & Cohn, S.E. (1999). Construction of correlation functions
  in two and three dimensions. *Quarterly Journal of the Royal
  Meteorological Society* 125:723–757. (Localization taper.)

Composition notes
-----------------
- ``models.pmcmc_sv`` (wave 24): likelihood-based Bayesian SV estimation —
  EKI is the likelihood-free ensemble complement.
- ``models.durbin_koopman`` (wave 27): Gaussian filtering/smoother for
  state space; EKI inverts static parameters without a likelihood.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
FORBIDDEN_KEYS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})

ForwardMap = Callable[[FloatArray], FloatArray]
PriorSampler = Callable[[int, np.random.Generator], FloatArray]


@dataclass(frozen=True)
class EKIResult:
    """EKI run output: posterior ensemble + per-iteration diagnostics."""

    theta_post: FloatArray  # (J, p) final ensemble
    theta_mean: FloatArray  # (p,) posterior mean
    misfit_path: FloatArray  # (n_iter,) mean Γ-weighted misfit
    spread_path: FloatArray  # (n_iter,) mean marginal std
    n_iter: int


def _check_obs(obs: object) -> FloatArray:
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 1 or not np.isfinite(arr).all():
        raise ValueError("obs must be a finite 1-D array")
    return arr


def _check_cov(cov: object, d: int) -> FloatArray:
    arr = np.asarray(cov, dtype=np.float64)
    if arr.shape != (d, d) or not np.isfinite(arr).all():
        raise ValueError(f"cov_obs must be finite ({d}, {d})")
    try:
        np.linalg.cholesky(arr + 1e-12 * np.eye(d))
    except np.linalg.LinAlgError as e:
        raise ValueError("cov_obs must be positive definite") from e
    return arr


def _check_ensemble(ens: object, name: str = "ensemble") -> FloatArray:
    arr = np.asarray(ens, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 4:
        raise ValueError(f"{name} must be a (J, p) array with J >= 4")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _run_forward(forward: ForwardMap, theta: FloatArray, d: int) -> FloatArray:
    """Evaluate the forward map on every ensemble member; fail closed."""
    g = np.empty((theta.shape[0], d))
    for j in range(theta.shape[0]):
        out = np.asarray(forward(theta[j]), dtype=np.float64)
        if out.shape != (d,) or not np.isfinite(out).all():
            raise ValueError(
                f"forward map must return finite (d={d},) — got shape {out.shape} at member {j}"
            )
        g[j] = out
    return g


def eki_update(
    ensemble: object,
    obs: object,
    forward: ForwardMap,
    cov_obs: object,
    rng: np.random.Generator,
    inflation: float = 1.0,
    perturb: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """One stochastic EKI update.

    ``θ_j⁺ = θ_j + C^{θg}(C^{gg} + Γ)⁻¹(y + η_j − G(θ_j))`` with sample
    covariances and ``η_j ~ N(0, Γ)`` (set ``perturb=False`` for the
    deterministic square-root-free variant). ``inflation > 1`` multiplies
    the ensemble's centered anomalies pre-update (bounded mean-reversion
    inflation, documented).
    """
    th = _check_ensemble(ensemble)
    y = _check_obs(obs)
    gamma = _check_cov(cov_obs, y.size)
    if not np.isfinite(inflation) or not 0 < inflation <= 4.0:
        raise ValueError("inflation must be in (0, 4]")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be np.random.Generator")

    n_j = th.shape[0]
    mean = th.mean(axis=0)
    th_inflated = mean + inflation * (th - mean) if inflation != 1.0 else th

    g = _run_forward(forward, th_inflated, y.size)
    g_mean = g.mean(axis=0)
    g_c = g - g_mean
    t_c = th_inflated - mean
    c_gg = (g_c.T @ g_c) / (n_j - 1)
    c_tg = (t_c.T @ g_c) / (n_j - 1)

    chol = np.linalg.cholesky(gamma)
    innov = g - y[None, :]
    if perturb:
        eta = (chol @ rng.standard_normal((y.size, n_j))).T
        target = innov - eta
    else:
        target = innov

    a = c_gg + gamma
    step = np.linalg.solve(a, target.T).T  # (J, d) solves for correction
    th_next = th_inflated - step @ c_tg.T
    if not np.isfinite(th_next).all():
        raise ValueError("non-finite ensemble after update — regularize Γ")
    return th_next, g


def ensemble_kalman_inversion(
    obs: object,
    prior_sampler: PriorSampler,
    forward: ForwardMap,
    n_ens: int = 60,
    n_iter: int = 25,
    seed: int = 0,
    inflation: float = 1.0,
    perturb: bool = True,
    cov_obs: object | None = None,
) -> EKIResult:
    """Full EKI loop: prior ensemble → posterior ensemble + history.

    ``prior_sampler(J, rng) -> (J, p)`` draws the initial ensemble;
    ``forward(theta_row) -> (d,)`` is the simulator; ``cov_obs`` is the
    observation-noise covariance Γ (default: identity — callers may fold
    the noise scale into the forward map instead). Records mean Γ-weighted
    misfit ``||y − G(θ_j)||_Γ`` and ensemble spread per iteration.
    """
    y = _check_obs(obs)
    if n_ens < 4 or n_iter < 1:
        raise ValueError("need n_ens >= 4 and n_iter >= 1")
    if not np.isfinite(inflation) or not 0 < inflation <= 4.0:
        raise ValueError("inflation must be in (0, 4]")
    rng = np.random.default_rng(seed)
    cov = np.eye(y.size) if cov_obs is None else _check_cov(cov_obs, y.size)
    th = _check_ensemble(prior_sampler(n_ens, rng), "prior sample")

    misfits = np.empty(n_iter)
    spreads = np.empty(n_iter)
    whiten = np.linalg.inv(np.linalg.cholesky(cov))
    for it in range(n_iter):
        g = _run_forward(forward, th, y.size)
        misfits[it] = float(np.mean(np.linalg.norm((g - y[None, :]) @ whiten.T, axis=1)))
        spreads[it] = float(th.std(axis=0).mean())
        th, _ = eki_update(th, y, forward, cov, rng, inflation=inflation, perturb=perturb)
    return EKIResult(
        theta_post=th,
        theta_mean=th.mean(axis=0),
        misfit_path=misfits,
        spread_path=spreads,
        n_iter=n_iter,
    )


def tikhonov_eki(
    obs: object,
    prior_sampler: PriorSampler,
    forward: ForwardMap,
    prior_mean: object,
    prior_cov: object,
    n_ens: int = 60,
    n_iter: int = 25,
    seed: int = 0,
) -> EKIResult:
    """EKI with Tikhonov prior regularization (augmented-forward variant).

    Appends the prior mean residual ``L·(θ − θ_0)`` as extra observations:
    ``G_aug(θ) = [G(θ); L·θ]``, ``y_aug = [y; L·θ_0]`` with ``L = chol(Σ₀⁻¹)``.
    Equivalent to penalizing ``||θ − θ_0||²_{Σ₀⁻¹}`` — the standard
    regularized-misfit EKI of Iglesias et al. (2013) §4.
    """
    y = _check_obs(obs)
    mu0 = np.asarray(prior_mean, dtype=np.float64)
    sig0 = np.asarray(prior_cov, dtype=np.float64)
    if mu0.ndim != 1 or sig0.shape != (mu0.size, mu0.size):
        raise ValueError("prior_mean (p,) and prior_cov (p, p) required")
    try:
        l_mat = np.linalg.cholesky(np.linalg.pinv(sig0))
    except np.linalg.LinAlgError as e:
        raise ValueError("prior_cov must be positive definite") from e

    def aug_forward(th_row: FloatArray) -> FloatArray:
        return np.concatenate([forward(th_row), l_mat @ th_row])

    y_aug = np.concatenate([y, l_mat @ mu0])
    return ensemble_kalman_inversion(
        y_aug,
        prior_sampler,
        aug_forward,
        n_ens=n_ens,
        n_iter=n_iter,
        seed=seed,
        cov_obs=np.eye(y_aug.size),
    )


def synth_inverse_problem(
    n_obs: int, n_params: int, seed: int = 0, noise_sd: float = 0.1
) -> dict[str, FloatArray]:
    """Noisy linear inverse problem: ``y = A θ_true + ε``, ε ~ N(0, σ²I)."""
    if n_obs < n_params or n_params < 1 or noise_sd <= 0:
        raise ValueError("need n_obs >= n_params >= 1 and noise_sd > 0")
    rng = np.random.default_rng(seed)
    a_mat = rng.normal(0.0, 1.0, (n_obs, n_params))
    theta_true = rng.normal(0.0, 1.0, n_params)
    y = a_mat @ theta_true + rng.normal(0.0, noise_sd, n_obs)
    return {
        "y": y,
        "a": a_mat,
        "theta_true": theta_true,
        "cov_obs": np.eye(n_obs) * noise_sd**2,
    }


def sv_moment_forward(theta: FloatArray, n_obs: int = 800, seed: int = 1000) -> FloatArray:
    """Moment-matching forward for a 1-factor stochastic-volatility DGP.

    ``θ = (raw_phi, raw_sigma_v)`` maps to ``φ = σ(raw_phi)`` logistic,
    ``σ_v = exp(raw)``; simulates ``h_{t+1} = φ h_t + σ_v η_t``,
    ``r_t = exp(h_t/2) ε_t`` and returns ``[acf_1, acf_5, excess_kurtosis]``
    of ``r``. Deterministic per member (fixed sim seed inside) so EKI sees
    a smooth map; obs should carry Monte-Carlo noise via cov_obs.
    """
    if theta.size != 2:
        raise ValueError("sv_moment_forward expects theta of size 2")
    phi = 1.0 / (1.0 + np.exp(-theta[0]))
    sigma_v = float(np.exp(np.clip(theta[1], -4.0, 2.0)))
    rng = np.random.default_rng(seed)
    h = np.empty(n_obs)
    h[0] = 0.0
    eta = rng.standard_normal(n_obs)
    eps = rng.standard_normal(n_obs)
    for t in range(1, n_obs):
        h[t] = phi * h[t - 1] + sigma_v * eta[t]
    r = np.exp(np.clip(h, -8.0, 8.0) / 2.0) * eps
    r_c = r - r.mean()
    var = float(r_c @ r_c) / r_c.size
    acf1 = float(r_c[1:] @ r_c[:-1]) / (r_c.size * var)
    acf5 = float(r_c[5:] @ r_c[:-5]) / (r_c.size * var)
    kurt = float(np.mean((r_c / np.sqrt(var)) ** 4) - 3.0)
    return np.array([acf1, acf5, kurt])


def sv_moment_obs(phi_true: float, sigma_v_true: float, seed: int) -> FloatArray:
    """Moment vector from the 'true' SV parameters."""
    raw = np.array([np.log(phi_true / (1.0 - phi_true)), np.log(sigma_v_true)])
    return sv_moment_forward(raw, seed=seed)


def bench_ensemble_kalman_inversion(seed: int = 0) -> dict[str, float]:
    """SYNTHETIC bench for EKI: linear recovery + SV moment matching."""
    rng0 = np.random.default_rng(seed)
    d = synth_inverse_problem(20, 4, seed=seed)
    a_mat, y, th_true = d["a"], d["y"], d["theta_true"]

    def lin_forward(th: FloatArray) -> FloatArray:
        return a_mat @ th

    def lin_prior(j: int, r: np.random.Generator) -> FloatArray:
        return r.normal(0.0, 2.0, (j, 4))

    res = ensemble_kalman_inversion(
        y,
        lin_prior,
        lin_forward,
        n_ens=80,
        n_iter=30,
        seed=seed,
        cov_obs=d["cov_obs"],
    )
    est = res.theta_mean
    param_relerr = float(np.linalg.norm(est - th_true) / np.linalg.norm(th_true))
    misfit_red = float(res.misfit_path[0] / max(res.misfit_path[-1], 1e-12))
    spread_collapse = float(res.spread_path[-1] / max(res.spread_path[0], 1e-12))

    # posterior coverage: fraction of true params inside ensemble 5–95%
    lo = np.quantile(res.theta_post, 0.05, axis=0)
    hi = np.quantile(res.theta_post, 0.95, axis=0)
    coverage = float(np.mean((th_true >= lo) & (th_true <= hi)))

    # SV moment matching: obs from truth φ=0.97, σ_v=0.25
    y_sv = sv_moment_obs(0.97, 0.25, seed=seed)

    def sv_prior(j: int, r: np.random.Generator) -> FloatArray:
        return np.column_stack([r.normal(3.0, 0.8, j), r.normal(-1.5, 0.5, j)])

    def sv_fwd(th: FloatArray) -> FloatArray:
        return sv_moment_forward(th, n_obs=600, seed=seed)

    res_sv = ensemble_kalman_inversion(
        y_sv,
        sv_prior,
        sv_fwd,
        n_ens=60,
        n_iter=20,
        seed=seed + 5,
        cov_obs=np.diag([0.05**2, 0.05**2, 0.5**2]),
    )
    m_err = float(np.linalg.norm(res_sv.theta_mean - np.array([np.log(0.97 / 0.03), np.log(0.25)])))

    tik = tikhonov_eki(
        y,
        lin_prior,
        lin_forward,
        prior_mean=np.zeros(4),
        prior_cov=np.eye(4) * 4.0,
        n_ens=80,
        n_iter=30,
        seed=seed,
    )
    tik_err = float(np.linalg.norm(tik.theta_mean - th_true) / np.linalg.norm(th_true))

    res2 = ensemble_kalman_inversion(
        y,
        lin_prior,
        lin_forward,
        n_ens=80,
        n_iter=30,
        seed=seed,
        cov_obs=d["cov_obs"],
    )
    determinism = float(np.array_equal(res2.theta_post, res.theta_post))
    _ = rng0

    blob: dict[str, float] = {
        "synthetic_param_relerr": param_relerr,
        "synthetic_tikhonov_relerr": tik_err,
        "synthetic_misfit_reduction": misfit_red,
        "synthetic_spread_collapse": spread_collapse,
        "synthetic_posterior_coverage": coverage,
        "synthetic_sv_param_err": m_err,
        "synthetic_sv_misfit_final": float(res_sv.misfit_path[-1]),
        "synthetic_sv_misfit_reduction": float(
            res_sv.misfit_path[0] / max(res_sv.misfit_path[-1], 1e-12)
        ),
        "synthetic_monotone_misfit_frac": float(np.mean(np.diff(res.misfit_path) <= 0)),
        "synthetic_determinism": determinism,
    }
    for k in blob:
        if FORBIDDEN_KEYS.intersection(k.split("_")):
            raise ValueError(f"forbidden bench key {k!r}")
    return blob
