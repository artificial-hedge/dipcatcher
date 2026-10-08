"""Doubly-robust policy value estimation (Dudík et al. 2011) (SYNTHETIC).

Contextual-bandit logging data with known propensity; a learned
target policy's value via IPS and DR vs the naive (regression) plug-in.
"""

from __future__ import annotations

import numpy as np


def bench_policy_value(
    seed: int = 583,
    n: int = 2000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 4))
    # logging policy: biased toward arm1 when x0>0
    e1 = 1 / (1 + np.exp(-(x[:, 0])))
    a = (rng.uniform(0, 1, n) < e1).astype(int)
    # rewards: arm1 better when x0<0 (opposite of logging bias)
    r1 = -x[:, 0] + 0.8 * np.maximum(x[:, 1], 0) - 0.4 * x[:, 1] ** 2
    r0 = np.zeros(n)
    y = np.where(a == 1, r1, r0) + rng.normal(0, 0.3, n)
    # target policy: always arm1
    target = np.ones(n, int)
    true_val = float(r1.mean())
    # direct model: fit y ~ x on arm1 rows
    from sklearn.linear_model import LinearRegression

    m1 = LinearRegression().fit(x[a == 1], y[a == 1])
    dm = float(m1.predict(x).mean())
    # IPS
    e_hat = np.clip(e1, 0.05, 0.95)
    ips = float(np.mean((target == a) * y / np.where(a == 1, e_hat, 1 - e_hat)))
    # DR
    m0 = LinearRegression().fit(x[a == 0], y[a == 0])
    mu_t = m1.predict(x)
    dr = float(
        np.mean(
            mu_t
            + (target == a)
            * (y - np.where(a == 1, m1.predict(x), m0.predict(x)))
            / np.where(a == 1, e_hat, 1 - e_hat)
        )
    )
    return {
        "synthetic_pv_dr_err": abs(dr - true_val),
        "synthetic_pv_dm_err": abs(dm - true_val),
        "synthetic_pv_ips_err": abs(ips - true_val),
        "synthetic_pv_dr_gain": abs(dm - true_val) - abs(dr - true_val),
    }
