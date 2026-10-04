"""Semblance velocity analysis on a CMP gather.

NMO trajectory t^2 = t0^2 + x^2/v^2; semblance S(v) = sum over a window of
(sum over traces)^2 normalized by trace energy. Velocity pick = argmax S(v).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 935


def hyperbolic_gather(
    events: list[tuple[float, float, float]],
    offsets: np.ndarray,
    dt: float,
    n: int,
    wavelet: np.ndarray,
) -> np.ndarray:
    """events: (t0, velocity, amplitude). Returns (n, n_traces) gather."""
    g = np.zeros((n, len(offsets)))
    for t0, v, amp in events:
        t = np.sqrt(t0**2 + (offsets / v) ** 2)
        for j, tj in enumerate(t):
            i = int(round(tj / dt))
            if 0 <= i < n:
                g[i, j] += amp
    for j in range(len(offsets)):
        g[:, j] = np.convolve(g[:, j], wavelet, mode="same")
    return g


def semblance(
    g: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    vel_grid: np.ndarray,
    t0: float,
    win: int = 15,
) -> float:
    n = g.shape[0]
    num = 0.0
    for v in vel_grid:
        num_v = 0.0
        den_v = 0.0
        t = np.sqrt(t0**2 + (offsets / v) ** 2)
        for k in range(-win // 2, win // 2 + 1):
            s = 0.0
            e = 0.0
            for j in range(len(offsets)):
                i = int(round(t[j] / dt)) + k
                if 0 <= i < n:
                    s += g[i, j]
                    e += g[i, j] ** 2
            num_v += s * s
            den_v += e
        if den_v * len(offsets) > 0:
            num += num_v / (den_v * len(offsets))
    return num


def semblance_scan(
    g: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    t0_grid: np.ndarray,
    vel_grid: np.ndarray,
    win: int = 15,
) -> np.ndarray:
    return np.array(
        [[semblance(g, dt, offsets, np.array([v]), t0, win) for v in vel_grid] for t0 in t0_grid]
    )


def _ricker(f: float, dt: float) -> np.ndarray:
    h = int(np.ceil(1.5 / (f * dt)))
    t = np.arange(-h, h + 1) * dt
    a = np.pi**2 * f**2 * t**2
    return (1.0 - 2.0 * a) * np.exp(-a)


def bench_semblance_scan(seed: int = _SEED) -> dict[str, float]:
    dt = 0.002
    n = 500
    offsets = np.linspace(50.0, 1500.0, 30)
    events = [(0.3, 2000.0, 1.0), (0.6, 2500.0, 0.8), (0.85, 3100.0, 0.7)]
    g = hyperbolic_gather(events, offsets, dt, n, _ricker(25.0, dt))
    vel_grid = np.linspace(1500.0, 3500.0, 81)
    hits = []
    for t0, v_true, _ in events:
        i0 = int(round(t0 / dt))
        num = np.zeros(len(vel_grid))
        den = np.zeros(len(vel_grid))
        for k, v in enumerate(vel_grid):
            t = np.sqrt(t0**2 + (offsets / v) ** 2)
            s2 = 0.0
            e = 0.0
            for di in range(-20, 21, 4):
                s = 0.0
                for j in range(len(offsets)):
                    i = i0 + int(round((t[j] - t0) / dt)) + di
                    if 0 <= i < n:
                        s += g[i, j]
                        e += g[i, j] ** 2
                s2 += s * s
            num[k] = s2
            den[k] = e * len(offsets)
        s_v = num / np.maximum(den, 1e-30)
        v_est = float(vel_grid[int(np.argmax(s_v))])
        hits.append(abs(v_est - v_true) / v_true < 0.05)
    return {"synthetic_semblance_scan": float(np.mean(hits))}
