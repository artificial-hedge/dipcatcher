"""Parks–McClellan equiripple FIR design via the Remez exchange.

Canonical reference: Parks & McClellan (1972); McClellan, Parks &
Rabiner (1973). At each step the extremal set pins the (n+1)×(n+1)
linear system  Σ_k a_k cos(k ω_i) + (−1)^i δ/W_i = D_i ; the dense
error is evaluated straight from the coefficients — no barycentric
interpolation — which keeps band edges and Nyquist pinned.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _extrema(E: FloatArray, n: int, edges: list[int]) -> FloatArray:
    """n+1 alternating-sign local maxima of |E| (classic exchange).

    Band edges are structural extrema and enter unconditionally."""
    aE = np.abs(E)
    cand = [0]
    cand += [i for i in range(1, len(E) - 1) if aE[i] >= aE[i - 1] and aE[i] >= aE[i + 1]]
    cand.append(len(E) - 1)
    cand = sorted(set(cand) | set(edges))
    # merge same-sign runs, keeping the larger |E|
    changed = True
    while changed:
        changed = False
        out = [cand[0]]
        for i in cand[1:]:
            if np.sign(E[i]) == np.sign(E[out[-1]]) or E[out[-1]] == 0:
                if aE[i] > aE[out[-1]]:
                    out[-1] = i
                changed = True
            else:
                out.append(i)
        cand = out
    # trim to n+1 by dropping the weaker end extremum
    while len(cand) > n + 1:
        cand = cand[1:] if aE[cand[0]] < aE[cand[-1]] else cand[:-1]
    return np.array(cand)


def _cosbasis(f: FloatArray, n: int) -> FloatArray:
    return np.cos(np.outer(2 * np.pi * f, np.arange(n)))


def remez(
    numtaps: int,
    bands: list[float],
    desired: list[float],
    weight: list[float] | None = None,
    grid_density: int = 16,
    max_iter: int = 60,
) -> FloatArray:
    """Equiripple linear-phase (type-I) FIR. bands in [0, 0.5]."""
    if numtaps % 2 == 0:
        raise ValueError("type-I symmetric design needs odd numtaps")
    nb = len(desired)
    if len(bands) != 2 * nb:
        raise ValueError("bands must pair with desired")
    if weight is None:
        weight = [1.0] * nb
    n = (numtaps + 1) // 2
    widths = np.array([bands[2 * i + 1] - bands[2 * i] for i in range(nb)])
    per = np.maximum(2, np.round(grid_density * n * widths / widths.sum()).astype(int))
    segs = [
        np.linspace(bands[2 * i], bands[2 * i + 1], per[i], endpoint=(i == nb - 1))
        for i in range(nb)
    ]
    grid = np.concatenate(segs)
    D = np.concatenate([[desired[i]] * len(segs[i]) for i in range(nb)])
    W = np.concatenate([[weight[i]] * len(segs[i]) for i in range(nb)])
    edges: list[int] = []
    off = 0
    for i in range(nb):
        edges += [off, off + len(segs[i]) - 1]
        off += len(segs[i])
    idx = np.round(np.linspace(0, len(grid) - 1, n + 1)).astype(int)
    a = np.zeros(n)
    for _ in range(max_iter):
        signs = (-1.0) ** np.arange(n + 1)
        M = np.column_stack([_cosbasis(grid[idx], n), signs / W[idx]])
        sol = np.linalg.solve(M, D[idx])
        a = sol[:n]
        E = W * (_cosbasis(grid, n) @ a - D)
        new_idx = _extrema(E, n, edges)
        if np.array_equal(new_idx, idx):
            break
        idx = new_idx
    h = np.zeros(numtaps)
    mid = numtaps // 2
    h[mid] = a[0]
    for k in range(1, n):
        h[mid - k] = a[k] / 2
        h[mid + k] = a[k] / 2
    return h


def freq_response(h: FloatArray, f: FloatArray) -> FloatArray:
    """Magnitude |H(e^{j2πf})| at normalized freqs."""
    n = np.arange(len(h))
    return np.abs(np.exp(-2j * np.pi * np.outer(f, n)) @ h)


def bench_remez(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: lowpass meets passband ripple / stopband attenuation."""
    h = remez(41, [0.0, 0.1, 0.15, 0.5], [1.0, 0.0], [1.0, 10.0])
    pass_dev = float(np.abs(freq_response(h, np.linspace(0.0, 0.1, 64)) - 1.0).max())
    stop_att = float(-20 * np.log10(freq_response(h, np.linspace(0.15, 0.5, 256)).max() + 1e-15))
    # weight ratio 1:10 ⇒ stop deviation ≈ pass deviation
    stop_dev = float(freq_response(h, np.linspace(0.15, 0.5, 256)).max())
    mid = len(h) // 2
    return {
        "synthetic_remez_pass_dev": pass_dev,
        "synthetic_remez_stop_dev": stop_dev,
        "synthetic_remez_stop_att": stop_att,
        "synthetic_remez_weight_ratio": float(pass_dev / (stop_dev + 1e-15)),
        "synthetic_remez_spec_met": float(pass_dev < 0.12 and stop_att > 28.0),
        "synthetic_remez_sym_err": float(np.abs(h - h[::-1]).max()),
        "synthetic_remez_dc_err": float(abs(freq_response(h, np.array([0.0]))[0] - 1.0)),
        "synthetic_remez_mid_tap": float(h[mid]),
    }
