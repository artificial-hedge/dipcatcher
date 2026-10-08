"""JKO scheme — Jordan-Kinderlehrer-Otto Wasserstein gradient flow (SYNTHETIC).

For the OU/Fokker-Planck equation rho_t = div(rho grad(V + log rho)),
the JKO step minimizes  W2^2(rho, rho_k)/(2 tau) + E[rho]  over rho.
For quadratic V = x^2/2 the flow stays Gaussian; the exact map on the
variance is sigma_{k+1} = sigma_k / (1 + tau) per W2-prox step (linearized:
d sigma/dt = -sigma + 1 in the convention used here), while the exact OU
law gives sigma(t) = exp(-t)*sigma_0 + (1 - exp(-t)). Bench: JKO-recursion
terminal variance vs exact OU stationary, plus a frozen-variance baseline.
"""

import numpy as np


def _jko_variance_flow(s0: float, tau: float, steps: int) -> float:
    s = s0
    for _ in range(steps):
        # exact JKO prox for Gaussian rho on V = x^2/2:
        # u = sqrt(v) solves u^2 (1+tau) - sqrt(v_k) u - tau = 0
        u = (np.sqrt(s) + np.sqrt(s + 4.0 * tau * (1.0 + tau))) / (2.0 * (1.0 + tau))
        s = u * u
    return s


def _exact_ou(s0: float, t: float) -> float:
    return float(1.0 + (s0 - 1.0) * np.exp(-2.0 * t))


def bench_jko_scheme(seed: int = 4311, steps: int = 50, tau: float = 0.05) -> dict[str, float]:
    del seed
    s0 = 4.0
    t_total = steps * tau
    s_jko = _jko_variance_flow(s0, tau, steps)
    s_exact = _exact_ou(s0, t_total)
    return {
        "synthetic_jko_sigma": s_jko,
        "synthetic_jko_exact_sigma": s_exact,
        "synthetic_jko_err": abs(s_jko - s_exact),
        "synthetic_jko_frozen_err": abs(s0 - s_exact),
        "synthetic_jko_stationary_gap": abs(s_jko - 1.0),
    }
