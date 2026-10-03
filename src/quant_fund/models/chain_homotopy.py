"""Chain homotopy: contiguous simplicial maps on the 1-simplex (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_chain_homotopy(seed: int = 0) -> float:
    """Verify d*H + H*d = g - f for f = id, g = collapse-to-v0 on [v0, v1].

    Chains: C0 = <v0, v1>, C1 = <e = [v0,v1]>, d1 = [[-1],[1]] (d e = v1 - v0).
    f0 = I_2, f1 = 1; g0 = [[1,1],[0,0]] (both verts -> v0), g1 = 0.
    H0: C0 -> C1 with H0(v0) = 0, H0(v1) = -e  => H0 = [[0, -1]].
    """
    checks = []
    d1 = np.array([[-1.0], [1.0]])
    f0 = np.eye(2)
    g0 = np.array([[1.0, 1.0], [0.0, 0.0]])
    h0 = np.array([[0.0, -1.0]])
    # C_0 level: d1 @ H0 = g0 - f0 (H_{-1} = 0)
    checks.append(bool(np.allclose(d1 @ h0, g0 - f0)))
    # C_1 level: g1 - f1 = -1 on e; H0 @ d1 = [[0,-1]] @ [[-1],[1]] = [-1]
    checks.append(bool(np.allclose(h0 @ d1, np.array([[-1.0]]))))
    # d1 d2 = 0 on the 2-simplex: edges (e01,e12,e02); d[012] = e01+e12-e02
    d2_tri = np.array([[1.0], [1.0], [-1.0]])
    d1_tri = np.array([[-1.0, 0.0, -1.0], [1.0, -1.0, 0.0], [0.0, 1.0, 1.0]])
    checks.append(bool(np.allclose(d1_tri @ d2_tri, 0)))
    # chain maps commute with boundary: g d = d g on the 1-simplex:
    # g0 @ d1 maps e -> g(v1 - v0) = v0 - v0 = 0; d(g1 e) = d(0) = 0
    checks.append(bool(np.allclose(g0 @ d1, 0)))
    # chain-homotopic maps agree on H0: (g0 - f0)z is a boundary (in im d1)
    z = np.array([[1.0], [1.0]])
    diff = (g0 - f0) @ z
    a = float(np.linalg.lstsq(d1, diff, rcond=None)[0][0, 0])
    checks.append(bool(np.allclose(d1 @ np.array([[a]]), diff)))
    # v0 + v1 is not itself a boundary: im d1 = span(-1, 1), z not parallel
    checks.append(not bool(np.allclose(z, d1 * (z[0, 0] / d1[0, 0]))))
    return float(sum(checks) / len(checks))


def bench_chain_homotopy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chain_homotopy": _bench_chain_homotopy(seed)}
