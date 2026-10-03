"""New-Keynesian Phillips curve: π_t = β E_t π_{t+1} + κ x_t."""

import numpy as np

_SEED = 20261231 + 753


def nkpc_irf(beta: float, kappa: float, x_path: np.ndarray) -> np.ndarray:
    """Solve π backward given output-gap path (terminal π=0)."""
    n = len(x_path)
    pi = np.zeros(n)
    for t in range(n - 2, -1, -1):
        pi[t] = beta * pi[t + 1] + kappa * x_path[t]
    return pi


def bench_nk_phillips(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    x = np.concatenate([np.ones(10) * 0.01, np.zeros(40)])
    pi = nkpc_irf(0.99, 0.1, x)
    # inflation positive during boom, decays to zero
    ok = float(pi[0] > 0 and pi[0] < 0.2 and abs(pi[-1]) < 1e-9)
    # closed form check: pi[0] = kappa*sum beta^t x_t
    expect = 0.1 * sum(0.99**t * x[t] for t in range(len(x)))
    return {"synthetic_nkpc_sign": ok, "synthetic_nkpc_exact": float(abs(pi[0] - expect) < 1e-12)}
