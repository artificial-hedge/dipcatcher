"""Linear-quadratic mean-field game (Lasry-Lions LQ-MFG).

Agent: dx = (a x + b u) dt + sigma dW,
cost E[int_0^H (0.5*(q x^2 + r u^2) + eta*(x - mbar)^2) dt].
The MFG couples a Riccati equation for the value scalar with the
mean-field ODE for mbar(t); solved by forward-backward fixed-point
iteration. Bench: fixed-point residual and distance to the decoupled
(eta=0) LQR gain as eta -> small is a sanity limit; main check is the
residual against self-consistency.
"""

import numpy as np

from quant_fund.models._mfg_synth import LQ_A, LQ_B, LQ_ETA, LQ_H, LQ_Q, LQ_R


def _riccati_flow(mbar: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward Riccati scalars s(t), g(t) given mean field mbar(t)."""
    T = len(mbar)
    s = np.zeros(T)
    g = np.zeros(T)
    for t in range(T - 2, -1, -1):
        ds = (2 * LQ_A * s[t + 1] + LQ_Q + 2 * LQ_ETA - (LQ_B**2 / LQ_R) * s[t + 1] ** 2) * dt
        dg = (
            LQ_A * g[t + 1] - (LQ_B**2 / LQ_R) * s[t + 1] * g[t + 1] - 2 * LQ_ETA * mbar[t + 1]
        ) * dt
        s[t] = s[t + 1] + ds
        g[t] = g[t + 1] + dg
    return s, g


def _mean_flow(s: np.ndarray, g: np.ndarray, dt: float, m0: float = 0.3) -> np.ndarray:
    """Forward mean-field ODE: dmbar = (a mbar - (b^2/r)(s mbar + g)) dt."""
    T = len(s)
    m = np.zeros(T)
    m[0] = m0
    for t in range(T - 1):
        u = -(LQ_B / LQ_R) * (s[t] * m[t] + g[t])
        m[t + 1] = m[t] + (LQ_A * m[t] + LQ_B * u) * dt
    return m


def bench_mfg_lq(seed: int = 4201, nt: int = 400, iters: int = 200) -> dict[str, float]:
    del seed
    dt = LQ_H / nt
    mbar = np.full(nt, 0.3)
    for _ in range(iters):
        s, g = _riccati_flow(mbar, dt)
        new = _mean_flow(s, g, dt)
        mbar = 0.5 * mbar + 0.5 * new
    s, g = _riccati_flow(mbar, dt)
    m2 = _mean_flow(s, g, dt)
    resid = float(np.max(np.abs(m2 - mbar)))
    # sanity: decoupled LQR gain limit when eta -> 0
    k_mfg = float(-(LQ_B / LQ_R) * s[0])
    s0 = 0.0
    for _ in range(2000):
        s0 += (2 * LQ_A * s0 + LQ_Q - (LQ_B**2 / LQ_R) * s0**2) * dt * 0.05
    k_lqr = float(-(LQ_B / LQ_R) * s0)
    return {
        "synthetic_mfg_resid": resid,
        "synthetic_mfg_gain": k_mfg,
        "synthetic_mfg_lqr_gain": k_lqr,
        "synthetic_mfg_gain_shift": abs(k_mfg - k_lqr),
        "synthetic_mfg_terminal": float(mbar[-1]),
    }
