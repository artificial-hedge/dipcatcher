"""Mixed logit — random-coefficients choice model via simulated MLE (SYNTHETIC).

Tastes vary across choosers: β_i = μ + σ·η_i with η standard
normal. Choice probabilities integrate over the taste
distribution, approximated by draws (simulated maximum
likelihood). Detects preference heterogeneity that a fixed
multinomial logit averages away.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure parameter recovery on
generated random-coefficient utilities — never market evidence.

References:
- McFadden, D., Train, K. (2000). Mixed MNL models for discrete
  response. *J. Applied Econometrics* 15, 447-470 — simulated
  likelihood and any-taste-distribution approximation.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*,
  2nd ed., ch. 6 — R=100-draw panel-smooth logit likelihood.
- Revelt, D., Train, K. (1998). Mixed logit with repeated
  choices. *Review of Economics and Statistics* 80.
- Bhat, C. R. (2001). Quasi-random maximum simulated likelihood
  estimation of the mixed multinomial logit model.
  *Transportation Research B* 35 — Halton draws.

Composition: pure numpy + scipy — fixed per-choosers' draws
(Halton-like shift via van der Corput) inside a smooth expected
likelihood, BFGS on (μ, log σ); deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm as _norm

FloatArray = NDArray[np.float64]


def _vdc(n: int, base: int = 2) -> FloatArray:
    """van der Corput quasi-random sequence in (0,1)."""
    idx = np.arange(1, n + 1, dtype=np.float64)
    out = np.zeros(n)
    f = 1.0 / base
    i = idx.copy()
    while np.any(i > 0):
        digit = i % base
        out += digit * f
        i = np.floor(i / base)
        f /= base
    return out


def mixed_logit_fit(
    choice: FloatArray,
    x: FloatArray,
    n_draws: int = 100,
    seed: int = 7,
) -> dict[str, float]:
    """Simulated MLE for mixed logit with one random coefficient.

    ``x`` is (n × n_alt × 1): one taste-varying attribute (the
    coefficient on it is random normal (μ, σ²)). Returns μ̂, σ̂ and
    the fixed-MNL reference."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 3 or xx.shape[2] != 1:
        raise ValueError("x must be (choosers × alternatives × 1)")
    n, n_alt, _ = xx.shape
    c = np.asarray(choice, dtype=np.float64).ravel()
    if c.size != n or n < 60:
        raise ValueError("choice length must match choosers, n>=60")
    if n_alt < 3 or n_alt > 20:
        raise ValueError("need 3..20 alternatives")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")
    if np.any(c < 0) or np.any(c > n_alt - 1):
        raise ValueError("choice ids out of range")
    if np.unique(c).size < 3:
        raise ValueError("need >=3 chosen alternatives observed")
    if np.ptp(xx) < 1e-9:
        raise ValueError("attribute must vary")
    if not (20 <= n_draws <= 400):
        raise ValueError("n_draws in 20..400")

    ci = c.astype(int)
    xa = xx[:, :, 0]
    rng = np.random.default_rng(seed)
    # quasi-random normals: van der Corput shifted uniform → normal icdf
    u = (_vdc(n_draws) + rng.uniform(0, 1)) % 1.0
    draws = _norm.ppf(np.clip(u, 1e-8, 1 - 1e-8))  # (R,)

    def neg_sim_ll(theta: FloatArray) -> float:
        mu, log_sig = theta[0], theta[1]
        sig = float(np.exp(np.clip(log_sig, -4, 2)))
        # p_i = (1/R) Σ_r exp(V_i(b_r)|chosen)/Σ_j exp(V_ij(b_r))
        # simulated probabilities: log((1/R) Σ_r p_i(b_r))
        p_acc = np.zeros(n)
        for r in range(n_draws):
            b = mu + sig * draws[r]
            v = b * xa
            v = v - v.max(axis=1, keepdims=True)
            p_acc += np.exp(v[np.arange(n), ci]) / np.exp(v).sum(axis=1)
        p_mean = np.clip(p_acc / n_draws, 1e-300, 1.0)
        return -float(np.sum(np.log(p_mean)))

    res = minimize(
        neg_sim_ll,
        np.array([0.0, -0.5]),
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-4},
    )
    mu_hat, log_sig_hat = float(res.x[0]), float(res.x[1])
    sig_hat = float(np.exp(np.clip(log_sig_hat, -4, 2)))

    # reference: plain MNL (σ=0)
    def mnl_ll(mu: float) -> float:
        v = mu * xa
        v = v - v.max(axis=1, keepdims=True)
        p = np.exp(v[np.arange(n), ci]) / np.exp(v).sum(axis=1)
        return float(np.sum(np.log(np.clip(p, 1e-300, 1.0))))

    res0 = minimize(lambda m: -mnl_ll(float(m[0])), [0.0], method="Nelder-Mead")
    mu_mnl = float(res0.x[0])
    ll_mnl = mnl_ll(mu_mnl)
    ll_mix = -neg_sim_ll(np.array([mu_hat, log_sig_hat]))

    return {
        "n": float(n),
        "n_alt": float(n_alt),
        "mu": mu_hat,
        "sigma": sig_hat,
        "mu_mnl": mu_mnl,
        "ll_mixed": ll_mix,
        "ll_mnl": ll_mnl,
        "lr_hetero": float(2 * (ll_mix - ll_mnl)),
    }


def synth_mixed_choice(
    n: int = 600,
    n_alt: int = 4,
    mu: float = 1.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Mixed-logit DGP: per-chooser coefficient b_i ~ N(μ, σ²) on a
    single attribute; type-1 EV idiosyncratic errors."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, n_alt, 1))
    b = rng.normal(mu, sigma, n)
    eps = rng.gumbel(0.0, 1.0, (n, n_alt))
    u = b[:, None] * x[:, :, 0] + eps
    choice = np.argmax(u, axis=1)
    return {
        "choice": choice.astype(np.float64),
        "x": x,
        "mu_true": np.array([mu]),
        "sigma_true": np.array([sigma]),
    }


def bench_mixed_logit(
    seed: int = 20261231 + 227,
) -> dict[str, float]:
    """Mixed-logit self-check: σ>0 heterogeneity recovered via
    simulated likelihood; MNL misses the dispersion.
    All ``synthetic_*``."""
    d = synth_mixed_choice(mu=1.0, sigma=1.2, seed=seed)
    out = mixed_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), seed=seed)
    d0 = synth_mixed_choice(mu=0.9, sigma=0.01, seed=seed + 1)
    out0 = mixed_logit_fit(np.asarray(d0["choice"]), np.asarray(d0["x"]), seed=seed)
    out_b = mixed_logit_fit(np.asarray(d["choice"]), np.asarray(d["x"]), seed=seed)

    s = float(out["sigma"])
    m = float(out["mu"])
    return {
        "synthetic_mu": m,
        "synthetic_mu_err": float(abs(m - 1.0)),
        "synthetic_sigma": s,
        "synthetic_sigma_err": float(abs(s - 1.2)),
        "synthetic_lr_hetero": float(out["lr_hetero"]),
        "synthetic_homog_sigma": float(out0["sigma"]),
        "synthetic_detects": float(
            s > 0.5 and abs(m - 1.0) < 0.4 and s > float(out0["sigma"]) * 1.5
        ),
        "synthetic_determinism": float(s == float(out_b["sigma"]) and m == float(out_b["mu"])),
    }
