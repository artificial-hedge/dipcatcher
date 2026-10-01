"""Simulation-based (likelihood-free) inference.

Two complementary schemes for posteriors over simulator parameters when
the likelihood is intractable:

1. ABC — Approximate Bayesian Computation: rejection sampling on a
   summary distance, and ABC-SMC (sequential Monte Carlo with tolerance
   schedule + perturbation kernel + importance reweighting) per
   Beaumont (2009) / Toni et al. (2009).
2. NRE — neural ratio estimation: a small logistic MLP trained to
   separate joint samples (θ, x) from marginal samples (θ, x') — the
   learned log-odds estimates the likelihood-to-evidence ratio
   r(θ|x) = p(x|θ)/p(x) (Hermans, Begy & Louppe 2020).

References
----------
- Beaumont, Cornuet, Marin & Robert (2009). ABC for population genetics.
  *Biometrika* 96(4) — arXiv-style review.
- Toni, Welch, Strelkowa, Ipsen & Stumpf (2009). ABC-SMC for systems
  biology. *Journal of the Royal Society Interface* 6.
- Hermans, Begy & Louppe (2020). Likelihood-free MCMC with amortized
  approximate ratio estimators. *ICML* — arXiv:1903.04057.
- Cranmer, Brehmer & Louppe (2020). The frontier of SBI. *PNAS* 117(48)
  — arXiv:1911.01429.

Honesty
-------
All evaluations are SYNTHETIC: MA(2) and g-and-k simulators with known
posterior-support parameters. Keys report posterior-mean error,
empirical coverage and effective sample size on simulated draws —
never real-data inference claims.

Composition notes
-----------------
- ``models/pmcmc_sv.py``: particle-MCMC on *tractable* likelihoods —
  this module is the likelihood-free counterpart.
- ``models/dp_mixture.py``: MCMC over mixture posteriors — different
  inferential machinery.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

FloatArray = NDArray[np.float64]
ArrayDict = dict[str, FloatArray | np.float64]
Simulator = Callable[[FloatArray, np.random.Generator], FloatArray]
Prior = Callable[[np.random.Generator, int], FloatArray]


def _check_sim(sim: Simulator, theta: FloatArray, rng: np.random.Generator) -> FloatArray:
    x = np.asarray(sim(theta, rng), dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("simulator returned empty output")
    return x


def _dist(x: FloatArray, y: FloatArray) -> float:
    return float(np.linalg.norm(x - y) / math.sqrt(x.size))


def abc_rej(
    sim: Simulator,
    prior: Prior,
    x_obs: FloatArray,
    tol: float,
    n_accept: int = 200,
    max_trials: int = 50_000,
    seed: int = 0,
) -> ArrayDict:
    """Rejection ABC: accept θ ~ prior when d(sim(θ), x_obs) < tol."""
    x_obs = np.asarray(x_obs, dtype=np.float64).ravel()
    if tol <= 0 or n_accept < 1 or max_trials < n_accept:
        raise ValueError("invalid ABC budget")
    rng = np.random.default_rng(seed)
    ths: list[FloatArray] = []
    trials = 0
    while len(ths) < n_accept and trials < max_trials:
        theta = np.asarray(prior(rng, 1), dtype=np.float64).ravel()
        x = _check_sim(sim, theta, rng)
        trials += 1
        if x.size == x_obs.size and _dist(x, x_obs) < tol:
            ths.append(theta)
    arr = np.stack(ths) if ths else np.zeros((0, 1))
    return {
        "theta": np.asarray(arr),
        "n_accept": np.float64(len(ths)),
        "accept_rate": np.float64(len(ths) / max(trials, 1)),
        "trials": np.float64(trials),
    }


def abc_smc(
    sim: Simulator,
    prior: Prior,
    x_obs: FloatArray,
    n_particles: int = 200,
    rounds: int = 4,
    tol_schedule: FloatArray | None = None,
    seed: int = 0,
) -> ArrayDict:
    """ABC-SMC: populations with decreasing tolerance, Gaussian
    perturbation kernel scaled to the population std (Toni et al. 2009).

    Returns particle matrix (n_particles, d), weights, per-round ESS and
    accept rates."""
    x_obs = np.asarray(x_obs, dtype=np.float64).ravel()
    if n_particles < 10 or rounds < 1:
        raise ValueError("invalid SMC budget")
    rng = np.random.default_rng(seed)
    eps = (
        np.asarray(tol_schedule, dtype=np.float64)
        if tol_schedule is not None
        else np.geomspace(
            _dist(x_obs * 2.0, x_obs), 0.05 * _dist(x_obs, np.zeros_like(x_obs)), rounds
        )
    )
    if eps.size != rounds or np.any(eps <= 0) or np.any(np.diff(eps) > 0):
        raise ValueError("tol_schedule must be positive and decreasing")
    # round 0: rejection against loose tolerance
    pop = abc_rej(sim, prior, x_obs, eps[0], n_particles, seed=seed)
    particles = np.asarray(pop["theta"])
    weights = np.ones(particles.shape[0]) / max(particles.shape[0], 1)
    ess_hist = np.zeros(rounds)
    acc_hist = np.zeros(rounds)
    ess_hist[0] = 1.0 / np.sum(weights**2) if weights.size else 0.0
    acc_hist[0] = float(pop["accept_rate"])
    dim = particles.shape[1] if particles.ndim == 2 else 1
    for r in range(1, rounds):
        if particles.shape[0] < 5:
            break
        sigma = np.std(particles, axis=0, ddof=1) + 1e-12
        new_p: list[FloatArray] = []
        new_w: list[float] = []
        trials = 0
        while len(new_p) < n_particles and trials < 20 * n_particles:
            i = int(rng.choice(particles.shape[0], p=weights / weights.sum()))
            cand = particles[i] + rng.standard_normal(dim) * sigma
            trials += 1
            x = _check_sim(sim, cand, rng)
            if x.size == x_obs.size and _dist(x, x_obs) < eps[r]:
                new_p.append(cand)
                # importance weight: prior / proposal mixture
                num = 1.0  # uniform prior density over the box
                den = float(
                    np.sum(
                        weights
                        * sstats.norm.pdf(cand[None, :], particles, sigma[None, :]).prod(axis=1)
                    )
                )
                new_w.append(num / max(den, 1e-300))
        particles = np.stack(new_p)
        weights = np.asarray(new_w)
        weights /= max(weights.sum(), 1e-300)
        ess_hist[r] = 1.0 / np.sum(weights**2)
        acc_hist[r] = len(new_p) / max(trials, 1)
    return {
        "particles": np.asarray(particles, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "ess": np.asarray(ess_hist, dtype=np.float64),
        "accept_rate": np.asarray(acc_hist, dtype=np.float64),
    }


def _mlp_forward(
    x: FloatArray, w1: FloatArray, b1: FloatArray, w2: FloatArray, b2: float
) -> FloatArray:
    """One-hidden-layer MLP, tanh + logistic output score."""
    h = np.tanh(x @ w1 + b1)
    return np.asarray(1.0 / (1.0 + np.exp(-(h @ w2 + b2))), dtype=np.float64)


def ratio_estimator(
    sim: Simulator,
    prior: Prior,
    theta_dim: int,
    n_train: int = 4000,
    hidden: int = 16,
    iters: int = 400,
    lr: float = 0.05,
    seed: int = 0,
) -> Callable[[FloatArray, FloatArray], float]:
    """NRE: train a tiny MLP to classify joint vs marginal (θ, x) pairs.

    Returns ``log_ratio(theta, x)`` estimating log p(x|θ) − log p(x).
    Features are standardised per-dimension on the training pool.
    """
    rng = np.random.default_rng(seed)
    ths = np.asarray(prior(rng, n_train), dtype=np.float64)
    if ths.ndim != 2 or ths.shape[1] != theta_dim:
        raise ValueError("prior must return (n, theta_dim)")
    xs = np.stack([_check_sim(sim, ths[i], rng) for i in range(n_train)])
    x_dim = xs.shape[1]
    # joint: (θ_i, x_i); marginal: (θ_i, x_perm(i))
    perm = rng.permutation(n_train)
    joint = np.column_stack([ths, xs])
    marg = np.column_stack([ths, xs[perm]])
    z = np.vstack([joint, marg])
    y = np.concatenate([np.ones(n_train), np.zeros(n_train)])
    mu, sd = z.mean(axis=0), z.std(axis=0) + 1e-12
    zs = (z - mu) / sd
    w1 = rng.standard_normal((theta_dim + x_dim, hidden)) * 0.5
    b1 = np.zeros(hidden)
    w2 = rng.standard_normal(hidden) * 0.5
    b2 = 0.0
    n = zs.shape[0]
    for _ in range(iters):
        p_hat = _mlp_forward(zs, w1, b1, w2, b2)
        # logistic loss gradients
        err = (p_hat - y) / n
        h = np.tanh(zs @ w1 + b1)
        gw2 = h.T @ err
        gb2 = float(err.sum())
        dh = np.outer(err, w2) * (1 - h**2)
        gw1 = zs.T @ dh
        gb1 = dh.sum(axis=0)
        w2 -= lr * gw2
        b2 -= lr * gb2
        w1 -= lr * gw1
        b1 -= lr * gb1

    def log_ratio(theta: FloatArray, x: FloatArray) -> float:
        v = np.concatenate([np.asarray(theta).ravel(), np.asarray(x).ravel()])
        zv = (v - mu) / sd
        p = float(_mlp_forward(zv[None, :], w1, b1, w2, b2)[0])
        p = min(max(p, 1e-9), 1 - 1e-9)
        return float(math.log(p / (1 - p)))

    return log_ratio


# --- synthetic simulators ----------------------------------------------------


def synth_ma2(theta: FloatArray, rng: np.random.Generator, t: int = 200) -> FloatArray:
    """MA(2): x_t = e_t + θ1 e_{t-1} + θ2 e_{t-2}; summary = sample
    autocovariances at lags 0,1,2."""
    th = np.asarray(theta, dtype=np.float64).ravel()
    if th.size != 2:
        raise ValueError("MA(2) needs 2 parameters")
    e = rng.standard_normal(t + 2)
    x = e[2:] + th[0] * e[1:-1] + th[1] * e[:-2]
    c = np.array([np.mean(x**2), np.mean(x[1:] * x[:-1]), np.mean(x[2:] * x[:-2])])
    return np.asarray(c, dtype=np.float64)


def synth_gk(theta: FloatArray, rng: np.random.Generator, n: int = 100) -> FloatArray:
    """g-and-k quantile simulator; summary = 8 quantiles of the sample."""
    a, b, g, k = np.asarray(theta, dtype=np.float64).ravel()
    z = rng.standard_normal(n)
    q = a + b * (1 + 0.8 * (1 - np.exp(-g * z)) / (1 + np.exp(-g * z))) * ((1 + z**2) ** k) * z
    return np.asarray(np.quantile(q, np.linspace(0.05, 0.95, 8)), dtype=np.float64)


def prior_ma2(rng: np.random.Generator, n: int) -> FloatArray:
    """Uniform prior on the invertible MA(2) region [-2,2]x[-1,1]."""
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1, 1, n)])


def bench_sbi(seed: int = 20261231 + 163) -> dict[str, float]:
    """SYNTHETIC likelihood-free inference validation."""
    # observed summary at true theta
    th_true = np.array([0.8, -0.3])
    x_obs = synth_ma2(th_true, np.random.default_rng(seed + 1), t=400)
    # ABC rejection
    rej = abc_rej(
        lambda th, r: synth_ma2(th, r, t=400),
        prior_ma2,
        x_obs,
        tol=0.05,
        n_accept=300,
        seed=seed,
    )
    th_hat = np.asarray(rej["theta"]).mean(axis=0)
    post_err = float(np.linalg.norm(th_hat - th_true))
    # ABC-SMC
    smc = abc_smc(
        lambda th, r: synth_ma2(th, r, t=400),
        prior_ma2,
        x_obs,
        n_particles=150,
        rounds=4,
        tol_schedule=np.array([0.25, 0.12, 0.07, 0.04]),
        seed=seed + 3,
    )
    pm = np.average(smc["particles"], axis=0, weights=smc["weights"])
    smc_err = float(np.linalg.norm(pm - th_true))
    # coverage: fraction of posterior draws inside 2x prior std box
    cover = float(np.mean(np.abs(smc["particles"] - th_true) < 0.5))
    # NRE: ratio at true theta should exceed a far theta's
    lr_fn = ratio_estimator(
        lambda th, r: synth_ma2(th, r, t=100),
        prior_ma2,
        theta_dim=2,
        n_train=2500,
        iters=300,
        seed=seed + 5,
    )
    lr_true = lr_fn(th_true, x_obs)
    lr_far = lr_fn(np.array([-1.5, 0.9]), x_obs)
    det = abc_rej(
        lambda th, r: synth_ma2(th, r, t=100),
        prior_ma2,
        x_obs,
        tol=0.5,
        n_accept=20,
        seed=9,
    )["theta"]
    det2 = abc_rej(
        lambda th, r: synth_ma2(th, r, t=100),
        prior_ma2,
        x_obs,
        tol=0.5,
        n_accept=20,
        seed=9,
    )["theta"]
    return {
        "synthetic_abc_posterior_err": post_err,
        "synthetic_abc_accept_rate": float(rej["accept_rate"]),
        "synthetic_smc_posterior_err": smc_err,
        "synthetic_smc_ess_final": float(np.asarray(smc["ess"])[-1]),
        "synthetic_smc_cover05": cover,
        "synthetic_nre_logr_true": float(lr_true),
        "synthetic_nre_logr_far": float(lr_far),
        "synthetic_nre_margin": float(lr_true - lr_far),
        "synthetic_determinism": float(np.array_equal(det, det2)),
    }
