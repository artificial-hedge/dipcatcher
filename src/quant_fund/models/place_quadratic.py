"""Analytical quadratic placement (wave 291).

Fixed cells on boundary; free cells minimize sum of squared wirelength
over nets — solve the linear system via conjugate-gradient-free direct
Gaussian elimination and check star-node sinks near fixed centroid.
"""

import numpy as np

_SEED = 20261231 + 834


def place(
    n_free: int, fixed: dict[int, tuple[float, float]], nets: list[list[int]]
) -> dict[int, tuple[float, float]]:
    # node ids 0..n_free-1 free; fixed ids are >= n_free keyed by dict
    A = np.zeros((n_free, n_free))
    bx = np.zeros(n_free)
    by = np.zeros(n_free)
    for net in nets:
        for u in net:
            for v in net:
                if u == v:
                    continue
                if u < n_free:
                    A[u, u] += 1.0
                    if v < n_free:
                        A[u, v] -= 1.0
                    else:
                        bx[u] += fixed[v][0]
                        by[u] += fixed[v][1]
    xs = np.linalg.solve(A, bx)
    ys = np.linalg.solve(A, by)
    return {i: (float(xs[i]), float(ys[i])) for i in range(n_free)}


def bench_place_quadratic(seed: int = _SEED) -> dict[str, float]:
    # 1 free cell connected to 4 corners → centroid (0.5, 0.5)
    pos = place(
        1,
        {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0), 4: (1.0, 1.0)},
        [[0, 1], [0, 2], [0, 3], [0, 4]],
    )
    ok = int(abs(pos[0][0] - 0.5) < 1e-6 and abs(pos[0][1] - 0.5) < 1e-6)
    # chain of 2 free cells between two pads: equal spacing at 1/3, 2/3
    pos = place(2, {2: (0.0, 0.0), 3: (3.0, 0.0)}, [[2, 0], [0, 1], [1, 3]])
    ok += int(abs(pos[0][0] - 1.0) < 1e-6 and abs(pos[1][0] - 2.0) < 1e-6)
    return {"synthetic_place": float(ok == 2)}
