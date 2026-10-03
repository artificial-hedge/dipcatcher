"""Wardrop user equilibrium — selfish routing on congested networks.

Latencies t_a(x) = free_flow·(1 + b·(x/cap)^p) (BPR form). Equilibrium
via projected flow shifts: at each step every OD commodity shifts flow
toward its cheapest path (all-or-nothing + MSA step-size), converging
to the Wardrop condition — used paths carry equal minimal latency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bpr_latency(
    x: FloatArray,
    free: FloatArray,
    cap: FloatArray,
    b: FloatArray | float = 0.15,
    p: float = 4.0,
) -> FloatArray:
    """Additive congestion form t_e = free_e + b_e·(x_e/cap_e)^p.

    (Standard BPR is the multiplicative special case b_e = b·free_e;
    the additive form also covers t = x-style links.)
    """
    return np.asarray(free + np.asarray(b) * (x / cap) ** p, dtype=np.float64)


def wardrop_solve(
    paths: list[list[int]],
    demand: float,
    free: FloatArray,
    cap: FloatArray,
    iters: int = 4000,
    b: FloatArray | float = 0.15,
    p: float = 4.0,
) -> FloatArray:
    """MSA equilibrium for a single OD pair over parallel paths.

    `paths` = list of edge-index lists (one per route). Returns flow
    per path.
    """
    n_paths = len(paths)
    f = np.full(n_paths, demand / n_paths)
    for k in range(iters):
        x = np.zeros(len(free))
        for i, edges in enumerate(paths):
            for e in edges:
                x[e] += f[i]
        t_edge = bpr_latency(x, free, cap, b, p)
        t_path = np.array([sum(t_edge[e] for e in edges) for edges in paths])
        # all-or-nothing assignment to cheapest path
        y = np.zeros(n_paths)
        y[int(np.argmin(t_path))] = demand
        step = 1.0 / (k + 2)  # MSA step
        f = (1 - step) * f + step * y
    return f


def path_latencies(
    f: FloatArray,
    paths: list[list[int]],
    free: FloatArray,
    cap: FloatArray,
    b: FloatArray | float = 0.15,
    p: float = 4.0,
) -> FloatArray:
    x = np.zeros(len(free))
    for i, edges in enumerate(paths):
        for e in edges:
            x[e] += f[i]
    t = bpr_latency(x, free, cap, b, p)
    return np.asarray([sum(t[e] for e in edges) for edges in paths])


def bench_wardrop(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: Pigou network — constant-latency edge vs congestible
    edge (t = x). Wardrop puts ALL flow on the congestible link
    (t = 1 = free alternative). Braess check: adding a free link
    raises equilibrium latency."""
    out: dict[str, float] = {}
    # Pigou: edge0 free=1 cap=inf-ish; edge1 t=x → free=0,b=1,cap=1,p=1
    free = np.array([1.0, 0.0])
    cap = np.array([1e9, 1.0])
    paths = [[0], [1]]
    f = wardrop_solve(paths, 1.0, free, cap, iters=4000, b=np.array([0.0, 1.0]), p=1.0)
    lat = path_latencies(f, paths, free, cap, b=np.array([0.0, 1.0]))
    out["synthetic_wardrop_pigou_cong_flow"] = float(f[1])
    out["synthetic_wardrop_pigou_lat_gap"] = float(
        abs(lat[0] - lat[1]) if f.min() > 1e-3 else lat[0] - lat[1]
    )
    out["synthetic_wardrop_pigou_ok"] = float(f[1] > 0.95)
    # Braess paradox: edges [s→v, s→w, v→w(zero), v→t, w→t]
    # t_sv = x, t_wt = x, t_sw = 1, t_vt = 1, t_vw = ε (cheap shortcut)
    free_b = np.array([0.0, 1.0, 1e-6, 1.0, 0.0])
    cap_b = np.array([1.0, 1e9, 1e9, 1e9, 1.0])
    # paths: [sv,vt]=0,3 ; [sv,vw,wt]=0,2,4 ; [sw,wt]=1,4
    paths_b = [[0, 3], [0, 2, 4], [1, 4]]
    b_b = np.array([1.0, 0, 0, 0, 1.0])  # t=x on s→v and w→t
    f_b = wardrop_solve(paths_b, 1.0, free_b, cap_b, iters=8000, b=b_b, p=1.0)
    lat_b = path_latencies(f_b, paths_b, free_b, cap_b, b=b_b)
    # remove the shortcut: compare latency without edge 2
    paths_nb = [[0, 3], [1, 4]]
    f_nb = wardrop_solve(paths_nb, 1.0, free_b, cap_b, iters=8000, b=b_b, p=1.0)
    lat_nb = path_latencies(f_nb, paths_nb, free_b, cap_b, b=b_b)
    out["synthetic_wardrop_braess_lat"] = float(lat_b.max())
    out["synthetic_wardrop_no_braess_lat"] = float(lat_nb.max())
    out["synthetic_wardrop_braess_paradox"] = float(lat_b.max() > lat_nb.max() + 1e-3)
    return out


if __name__ == "__main__":
    print(bench_wardrop())
