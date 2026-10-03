"""Linear tau-p (slant-stack) transform.

tau(p) = t - p*x summed over offsets. Maps linear moveout events in
(t, x) to points in (tau, p). Seismic canon.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 891


def slant_stack(data: np.ndarray, x: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Slant stack of a (nt, nx) gather -> (nt, np) tau-p panel.

    For each p, shifts each trace by -p*x and sums over x.
    """
    d = np.asarray(data, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    dt = float(t[1] - t[0])
    nt = t.size
    out = np.zeros((nt, p.size))
    for j, pj in enumerate(p):
        acc = np.zeros(nt)
        for k, xk in enumerate(x):
            shift = pj * xk / dt
            i0 = int(np.floor(shift))
            frac = shift - i0
            # acc[tau] += d(tau + p x) = (1-frac)*d[tau+i0] + frac*d[tau+i0+1]
            idx = np.arange(nt) + i0
            valid = (idx >= 0) & (idx < nt)
            acc[valid] += (1 - frac) * d[idx[valid], k]
            idx2 = idx + 1
            valid2 = (idx2 >= 0) & (idx2 < nt)
            acc[valid2] += frac * d[idx2[valid2], k]
        out[:, j] = acc
    return out


def bench_taup_transform(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    nt, nx = 400, 30
    dt, dx = 0.004, 40.0
    t = np.arange(nt) * dt
    x = np.arange(nx) * dx
    # two linear events: t = tau_i + p_i x
    taus = np.array([0.5, 1.0])
    ps = np.array([2.0e-4, 5.0e-4])  # s/m
    data = rng.normal(0, 0.02, (nt, nx))
    for tau_i, p_i in zip(taus, ps, strict=True):
        for k in range(nx):
            it = int(round((tau_i + p_i * x[k]) / dt))
            if 0 <= it < nt:
                data[it, k] += 3.0
    p_axis = np.linspace(-0.001, 0.001, 200)
    panel = slant_stack(data, x, t, p_axis)
    score = 0.0
    found = []
    for tau_i, p_i in zip(taus, ps, strict=True):
        pmask = np.abs(p_axis - p_i) < 1e-4
        sub = np.abs(panel[:, pmask])
        it_loc, jp_loc = np.unravel_index(np.argmax(sub), sub.shape)
        p_est = p_axis[pmask][jp_loc]
        t_est = t[it_loc]
        found.append((abs(t_est - tau_i) < 3 * dt, abs(p_est - p_i) < 2.5e-5))
    score += sum(1.0 for a, b in found if a and b)
    # directivity: coherent stacking yields strong peak vs median panel level
    env = np.abs(panel)
    score += 1.0 if float(np.max(env) / (np.median(env) + 1e-12)) > 30.0 else 0.0
    return {"synthetic_taup_transform": score / 3.0}
