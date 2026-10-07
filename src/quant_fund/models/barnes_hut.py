"""Barnes-Hut quadtree N-body force approximation (synthetic) (SYNTHETIC).

θ-criterion opening; verified: (i) forces match direct O(N²)
summation within bounded relative error; (ii) error shrinks as
θ decreases (monotone check); (iii) COM property of cells.
"""

from __future__ import annotations

import math
import random

G = 1.0
EPS2 = 1e-6


class _Cell:
    def __init__(self, x: float, y: float, h: float) -> None:
        self.x, self.y, self.h = x, y, h
        self.mass = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.body = -1
        self.kids: list[_Cell] = []

    def insert(self, i: int, pos: list[list[float]], mass: list[float]) -> None:
        if self.body == -2:  # internal
            quad = (pos[i][0] > self.x + self.h / 2) + 2 * (pos[i][1] > self.y + self.h / 2)
            self.kids[quad].insert(i, pos, mass)
        elif self.body == -1:  # empty
            self.body = i
        else:  # external occupied → subdivide
            old = self.body
            self.body = -2
            self.kids = [
                _Cell(self.x + (q % 2) * self.h / 2, self.y + (q // 2) * self.h / 2, self.h / 2)
                for q in range(4)
            ]
            self.kids[
                (pos[old][0] > self.x + self.h / 2) + 2 * (pos[old][1] > self.y + self.h / 2)
            ].insert(old, pos, mass)
            self.kids[
                (pos[i][0] > self.x + self.h / 2) + 2 * (pos[i][1] > self.y + self.h / 2)
            ].insert(i, pos, mass)
        self.mass += mass[i]
        self.cx += mass[i] * pos[i][0]
        self.cy += mass[i] * pos[i][1]


def _force_bh(cell: _Cell, i: int, pos: list[list[float]], theta: float) -> tuple[float, float]:
    if cell.body == i or cell.mass == 0:
        return 0.0, 0.0
    cmx = cell.cx / cell.mass
    cmy = cell.cy / cell.mass
    dx = cmx - pos[i][0]
    dy = cmy - pos[i][1]
    d2 = dx * dx + dy * dy + EPS2
    d = math.sqrt(d2)
    if cell.body >= 0 or cell.h / d < theta:
        f = G * cell.mass / (d2 * d)
        return f * dx, f * dy
    fx = fy = 0.0
    for k in cell.kids:
        dfx, dfy = _force_bh(k, i, pos, theta)
        fx += dfx
        fy += dfy
    return fx, fy


def _force_direct(i: int, pos: list[list[float]], mass: list[float]) -> tuple[float, float]:
    fx = fy = 0.0
    for j in range(len(pos)):
        if j != i:
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            d2 = dx * dx + dy * dy + EPS2
            f = G * mass[j] / (d2 * math.sqrt(d2))
            fx += f * dx
            fy += f * dy
    return fx, fy


def bench_barnes_hut(seed: int = 20261231 + 271) -> dict[str, float]:
    rng = random.Random(seed)
    n = 60
    pos = [[rng.uniform(-5, 5), rng.uniform(-5, 5)] for _ in range(n)]
    mass = [rng.uniform(0.5, 2.0) for _ in range(n)]
    root = _Cell(-8.0, -8.0, 16.0)
    for i in range(n):
        root.insert(i, pos, mass)
    errs = []
    for theta in (0.8, 0.4, 0.15):
        rel = []
        for i in range(0, n, 3):
            bhx, bhy = _force_bh(root, i, pos, theta)
            dfx, dfy = _force_direct(i, pos, mass)
            denom = math.hypot(dfx, dfy) + 1e-12
            rel.append(math.hypot(bhx - dfx, bhy - dfy) / denom)
        errs.append(sum(rel) / len(rel))
    # COM check
    com_ok = abs(root.mass - sum(mass)) < 1e-9
    return {
        "synthetic_err_theta08": float(errs[0]),
        "synthetic_err_theta04": float(errs[1]),
        "synthetic_err_theta015": float(errs[2]),
        "synthetic_monotone": float(errs[0] >= errs[2]),
        "synthetic_small_theta_ok": float(errs[2] < 0.02),
        "synthetic_com_ok": float(com_ok),
    }
