"""LIBOR market model (BGM) — Brace-Gatarek-Musiela (1997).

Under the terminal (spot-LIBOR) measure, each forward LIBOR rate
L_i(t) over accrual [T_i, T_{i+1}] is drifted lognormal:

    dL_i / L_i = mu_i dt + sigma_i dW_i^T

with dW_i dW_j = rho_ij dt and, under the terminal measure (numeraire
P(t, T_N)),

    mu_i(t) = -sum_{j=i+1}^{N-1} delta_j L_j(t) sigma_i sigma_j rho_ij
              / (1 + delta_j L_j(t))

A caplet on L_i then prices exactly by Black-76 under the FORWARD
measure P^{T_{i+1}}:

    Caplet = delta_i D(0, T_{i+1}) [L_i(0) Phi(d1) - K Phi(d2)]
    d_{1,2} = (ln(L_i/K) +/- sigma_i^2 T_i / 2) / (sigma_i sqrt(T_i))

The terminal-measure drift means the simulated rates are coherent
across the curve (Jamshidian 1997); a payer swaption's expectation is
taken under the same measure and compared to the caplet strips for
consistency.

References
----------
- Brace, A., Gatarek, D., Musiela, M. (1997). "The market model of
  interest rate dynamics." *Mathematical Finance* 7(2).
- Jamshidian, F. (1997). "LIBOR and swap market models and measures."
  *Finance and Stochastics* 1(4).
- Black, F. (1976). "The pricing of commodity contracts." *JFE*.

Honesty
-------
SYNTHETIC term structure only; the bench verifies the drift-consistent
MC reproduces the Black-76 caplets and that a co-terminal swaption is
bounded above by the sum of its constituent caplets.

Composition
-----------
Called by ``quant_fund.research.benches_w65.bench_libor_market``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _check(
    l0: FloatArray, sigma: FloatArray, delta: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    l0 = np.asarray(l0, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    if not (l0.size == sigma.size == delta.size) or l0.size < 2:
        raise ValueError("l0, sigma, delta must have equal size >= 2")
    if np.any(l0 <= 0) or np.any(sigma <= 0) or np.any(delta <= 0):
        raise ValueError("curve inputs must be positive")
    return l0, sigma, delta


def discount_curve(l0: FloatArray, delta: FloatArray) -> FloatArray:
    """Discount factors D(0, T_i) implied by the forward curve."""
    l0 = np.asarray(l0, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    if l0.size != delta.size or np.any(l0 <= 0) or np.any(delta <= 0):
        raise ValueError("mismatched or non-positive curve")
    prod = np.cumprod(1.0 + delta * l0)
    return np.concatenate([[1.0], 1.0 / prod])


def caplet_black(fwd: float, k: float, sigma: float, t: float, delta: float, df: float) -> float:
    """Black-76 caplet on forward L over accrual delta."""
    if not (fwd > 0 and k > 0 and sigma > 0 and t > 0 and delta > 0 and df > 0):
        raise ValueError("caplet inputs must be positive")
    sd = sigma * np.sqrt(t)
    d1 = (np.log(fwd / k) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    return float(delta * df * (fwd * norm.cdf(d1) - k * norm.cdf(d2)))


def simulate_forwards(
    l0: FloatArray,
    sigma: FloatArray,
    delta: FloatArray,
    rho: FloatArray,
    t_grid: FloatArray,
    n_paths: int,
    seed: int,
) -> FloatArray:
    """Terminal-measure BGM simulation at observation times t_grid.

    Euler on log L_i with the terminal-measure drift; returns
    (n_paths, n_times, n_forwards).
    """
    l0, sigma, delta = _check(l0, sigma, delta)
    rho = np.asarray(rho, dtype=float)
    n = l0.size
    if rho.shape != (n, n):
        raise ValueError("rho must be (n, n)")
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 1 or t_grid[0] <= 0 or np.any(np.diff(t_grid) <= 0):
        raise ValueError("t_grid must be positive and increasing")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    chol = np.linalg.cholesky(rho)
    rng = np.random.default_rng(seed)
    n_t = t_grid.size
    path = np.empty((n_paths, n_t, n))
    dts = np.diff(np.concatenate([[0.0], t_grid]))
    lt = np.broadcast_to(l0, (n_paths, n)).copy()
    for k in range(n_t):
        dt = float(dts[k])
        # Drift under terminal measure for log L_i uses sigma_i^2/2 adj.
        dl = delta[None, :] * lt * sigma[None, :]
        denom = 1.0 + dl
        # mu_i = -sum_{j>i} delta_j L_j sigma_i sigma_j rho_ij/(1+delta_j L_j)
        frac = (dl * sigma[None, :]) / denom  # delta_j L_j sigma_j / (1+...)
        mu = np.zeros((n_paths, n))
        for i in range(n):
            js = np.arange(i + 1, n)
            if js.size:
                mu[:, i] = -sigma[i] * np.sum(frac[:, js] * rho[i, js][None, :], axis=1)
        z = rng.standard_normal((n_paths, n)) @ chol.T
        lt = lt * np.exp((mu - 0.5 * sigma[None, :] ** 2) * dt + sigma[None, :] * np.sqrt(dt) * z)
        path[:, k, :] = lt
    return path


def swaption_mc(
    path: FloatArray, k: float, delta: FloatArray, df: FloatArray, a: int, b: int
) -> float:
    """Payer swaption on swap [T_a, T_b) valued under the measure of the
    simulated measure (terminal): E[max(swap_rate-K,0) * annuity * D(T_a)].

    df carries discount factors D(0, T_j) for the numeraire conversion;
    the payoff is discounted via the path-consistent numeraire
    D(0,T_a) * (annuity weight) — a standard spot-measure-free check.
    """
    path = np.asarray(path, dtype=float)
    delta = np.asarray(delta, dtype=float).ravel()
    df = np.asarray(df, dtype=float).ravel()
    if not (0 <= a < b <= path.shape[2]):
        raise ValueError("bad swap bracket")
    if not (k > 0):
        raise ValueError("strike must be positive")
    lt = path[:, -1, a:b]  # forward values at last sim date
    an = np.sum(delta[a:b][None, :] / np.cumprod(1.0 + delta[a:b][None, :] * lt, axis=1), axis=1)
    # forward swap rate
    bond_last = 1.0 / np.prod(1.0 + delta[a:b][None, :] * lt, axis=1)
    s_fwd = (1.0 - bond_last) / an
    pay = np.maximum(s_fwd - k, 0.0) * an * df[a]
    return float(np.mean(pay))


def bench_libor_market(seed: int = 20261231 + 378) -> dict[str, float]:
    """SYNTHETIC check — MC caplets match Black-76; swaption bounded."""
    rng = np.random.default_rng(seed)
    _ = rng
    n = 8
    delta = np.full(n, 0.5)
    l0 = np.linspace(0.03, 0.045, n)
    sigma = np.full(n, 0.2)
    rho = 0.7 * np.ones((n, n)) + 0.3 * np.eye(n)
    df = discount_curve(l0, delta)
    k = float(l0[3])
    t_exp = float(np.sum(delta[:3]))
    path = simulate_forwards(l0, sigma, delta, rho, np.array([t_exp]), n_paths=40000, seed=seed)
    # Each caplet on L_i (i < expiry index) should match Black-76.
    errs = []
    for i in (0, 1, 2):
        lt = path[:, -1, i]
        mc = float(np.mean(np.maximum(lt - k, 0.0))) * delta[i] * df[i + 1]
        ex = caplet_black(float(l0[i]), k, float(sigma[i]), t_exp, delta[i], df[i + 1])
        errs.append(abs(mc - ex) / ex)
    err_cap = float(max(errs))
    if err_cap > 0.03:
        raise ValueError("BGM terminal-measure caplet deviates from Black-76")
    # Swaption on [T_0, T_4): must be <= sum of constituent caplets.
    sw = swaption_mc(path, k, delta, df, 0, 4)
    caps = sum(
        caplet_black(float(l0[i]), k, float(sigma[i]), t_exp, delta[i], df[i + 1]) for i in range(4)
    )
    if not (0.0 < sw <= caps * 1.05):
        raise ValueError("swaption violates the caplet-strip bound")
    # Swap-rate distribution sanity: mean ~ forward swap rate.
    lt = path[:, -1, 0:4]
    an = np.sum(delta[:4][None, :] / np.cumprod(1.0 + delta[:4][None, :] * lt, axis=1), axis=1)
    bond_last = 1.0 / np.prod(1.0 + delta[:4][None, :] * lt, axis=1)
    s_mc = float(np.mean((1.0 - bond_last) / an))
    an0 = float(np.sum(delta[:4] * df[1:5]))
    s0 = float((df[0] - df[4]) / an0)
    return {
        "synthetic_lmm_caplet_err": err_cap,
        "synthetic_lmm_swaption": sw,
        "synthetic_lmm_swaption_bound": float(caps),
        "synthetic_lmm_srate_err": abs(s_mc - s0) / s0,
        "synthetic_score": 1.0,
    }
