"""Continuous-time quantum walk on a cycle graph vs classical random
walk — hitting-probability at antipode after O(N) time (quadratic
speedup signature).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def bench_quantum_walk(seed: int = 3085, n: int = 20) -> dict[str, float]:
    # cycle adjacency
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[i, (i - 1) % n] = 1
    # quantum walk: |psi(t)> = exp(-i A t)|0>
    psi0 = np.zeros(n)
    psi0[0] = 1.0
    t = n / 2.0
    psi = np.asarray(expm(-1j * A * t) @ psi0)
    p_antipode_q = float(np.abs(psi[n // 2]) ** 2)
    # classical RW after same number of steps
    P = A / 2.0
    dist = np.zeros(n)
    dist[0] = 1.0
    for _ in range(int(t)):
        dist = P @ dist
    p_antipode_c = float(dist[n // 2])
    # quantum max over time grid (fast oscillation reaches antipode)
    probs_t = []
    for tt in np.linspace(1, n, 50):
        ps = expm(-1j * A * tt) @ psi0
        probs_t.append(float(np.abs(ps[n // 2]) ** 2))
    return {
        "synthetic_qwalk_p_antipode": p_antipode_q,
        "synthetic_qwalk_p_peak": float(np.max(probs_t)),
        "synthetic_cwalk_p_antipode": p_antipode_c,
        "synthetic_qwalk_gain": float(np.max(probs_t) - p_antipode_c),
        "torch_available": 0.0,
    }
