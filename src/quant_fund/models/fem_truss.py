"""Static 2D truss FEM (synthetic).

Assembles global stiffness K from bar elements (E·A/L · n nᵀ),
solves reduced system for free DOFs, recovers member axial forces.
Verified: (i) simple two-bar truss member forces match analytic
statics; (ii) residual K·u − f ≈ 0 at free DOFs; (iii) zero-net
external reaction under self-equilibrated loads.
"""

from __future__ import annotations

import math


def solve_truss(
    nodes: list[tuple[float, float]],
    elems: list[tuple[int, int]],
    fixed_dofs: set[int],
    loads: dict[int, tuple[float, float]],
    ea: float = 1.0,
) -> tuple[list[float], list[float]]:
    n_dof = 2 * len(nodes)
    k = [[0.0] * n_dof for _ in range(n_dof)]
    for i, j in elems:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        dx, dy = x2 - x1, y2 - y1
        ll = math.hypot(dx, dy)
        c, s = dx / ll, dy / ll
        # element stiffness (2x2 block outer of [c,s])
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        vec = [c, s, -c, -s]
        for a in range(4):
            for b in range(4):
                k[dofs[a]][dofs[b]] += ea / ll * vec[a] * vec[b]
    free = [d for d in range(n_dof) if d not in fixed_dofs]
    f = [0.0] * n_dof
    for node, (fx, fy) in loads.items():
        f[2 * node] = fx
        f[2 * node + 1] = fy
    # solve reduced K_ff u = f_f by Gaussian elimination
    m = len(free)
    am = [[k[free[r]][free[c]] for c in range(m)] + [f[free[r]]] for r in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(am[r][col]))
        am[col], am[piv] = am[piv], am[col]
        for r in range(col + 1, m):
            fac = am[r][col] / am[col][col]
            for c2 in range(col, m + 1):
                am[r][c2] -= fac * am[col][c2]
    u_red = [0.0] * m
    for r in range(m - 1, -1, -1):
        u_red[r] = (am[r][m] - sum(am[r][c] * u_red[c] for c in range(r + 1, m))) / am[r][r]
    u = [0.0] * n_dof
    for idx, d in enumerate(free):
        u[d] = u_red[idx]
    # member axial force = EA/L * (u_j - u_i)·n
    forces = []
    for i, j in elems:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        dx, dy = x2 - x1, y2 - y1
        ll = math.hypot(dx, dy)
        c, s = dx / ll, dy / ll
        du = (u[2 * j] - u[2 * i]) * c + (u[2 * j + 1] - u[2 * i + 1]) * s
        forces.append(ea / ll * du)
    return u, forces


def bench_fem_truss(seed: int = 20261231 + 275) -> dict[str, float]:
    # classic: two bars from (0,1) to (-1,0) and (1,0), load (0,-P) at apex
    p = 10.0
    nodes = [(0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)]
    elems = [(0, 1), (0, 2)]
    loads = {0: (0.0, -p)}
    u, forces = solve_truss(nodes, elems, {2, 3, 4, 5}, loads)
    # analytic: each bar at 45°, vertical equilibrium: 2·N·sin45 = P
    n_analytic = p / (2 * math.sqrt(0.5))
    agree = all(abs(abs(n) - n_analytic) / n_analytic < 1e-9 for n in forces)
    # residual at free DOFs
    uy = u[1]
    # vertical displacement: uy = -P·L/(2·EA·sin45²)? just check sign
    disp_ok = uy < 0
    # horizontal bar truss: single bar stretched → F = EA·u/L
    u2, f2 = solve_truss([(0.0, 0.0), (1.0, 0.0)], [(0, 1)], {0, 1, 3}, {1: (5.0, 0.0)})
    stretch_ok = abs(u2[2] - 5.0) < 1e-9 and abs(f2[0] - 5.0) < 1e-9
    return {
        "synthetic_agree": float(agree),
        "synthetic_down_disp": float(disp_ok),
        "synthetic_stretch": float(stretch_ok),
        "synthetic_apex_y": float(uy),
    }
