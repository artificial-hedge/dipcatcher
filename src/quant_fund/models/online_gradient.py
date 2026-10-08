"""Online gradient descent regret (Zinkevich O(sqrt(T))) (SYNTHETIC).

Adversarial convex losses l_t(x) = (x - a_t)^2 with drifting a_t on a
bounded interval; OGD with eta_t = D/(G sqrt(t)). Bench reports the
regret vs the best fixed action in hindsight and vs the sqrt(T) bound.
"""

import numpy as np


def bench_online_gradient(seed: int = 5411) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    T = 2000
    a = np.cumsum(rng.normal(0, 0.3, T)) % 6.0
    D, G = 6.0, 12.0
    x = 0.0
    tot = 0.0
    for t in range(1, T + 1):
        tot += (x - a[t - 1]) ** 2
        grad = 2.0 * (x - a[t - 1])
        x = np.clip(x - (D / (G * np.sqrt(t))) * grad, 0.0, 6.0)
    xs = np.linspace(0, 6, 121)
    hind = min(float(((xs_i - a) ** 2).sum()) for xs_i in xs)
    regret = tot - hind
    return {
        "synthetic_ogd_regret": regret,
        "synthetic_ogd_sqrt_T": float(np.sqrt(T)),
        "synthetic_ogd_regret_per_sqrt": regret / np.sqrt(T),
        "synthetic_ogd_loss": tot,
        "synthetic_ogd_hindsight": hind,
    }
