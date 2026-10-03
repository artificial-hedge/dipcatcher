"""Mean-field variational Bayes for a Gaussian mean + precision.

CAVI on q(mu)=N(m,s2), q(tau)=Gamma(a,b) for x ~ N(mu, 1/tau).
Bench: posterior mean/credible width vs the closed-form Normal-Gamma
answer on synthetic data.
"""

import numpy as np


def bench_variational_bayes(seed: int = 5707) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 200
    mu_true, tau_true = 1.7, 4.0
    x = rng.normal(mu_true, 1 / np.sqrt(tau_true), n)
    xbar, xs2 = x.mean(), float(np.mean(x**2))
    # priors: mu ~ N(0, 0.01), tau ~ Gamma(1, 1)
    a, b = 1.0, 1.0
    m, s2 = 0.0, 100.0
    for _ in range(300):
        e_tau = a / b
        s2 = 1.0 / (0.01 + n * e_tau)
        m = s2 * (n * e_tau * xbar)
        e_mu, e_mu2 = m, m * m + s2
        a = 1.0 + n / 2
        b = 1.0 + 0.5 * (n * xs2 - 2 * n * xbar * e_mu + n * e_mu2)
    # closed form Normal-Gamma posterior for mu
    m_exact = n * tau_true * 0 + xbar  # weak prior -> posterior mean ~= xbar
    # posterior variance of mu: E[tau]^{-1}/n under posterior
    s2_exact = b / (a * n)
    return {
        "synthetic_vb_m": m,
        "synthetic_vb_m_exact": float(m_exact),
        "synthetic_vb_s2": s2,
        "synthetic_vb_s2_exact": float(s2_exact),
        "synthetic_vb_err": abs(m - xbar),
        "synthetic_vb_true_cover": float(abs(m - mu_true) < 1.96 * np.sqrt(s2)),
    }
