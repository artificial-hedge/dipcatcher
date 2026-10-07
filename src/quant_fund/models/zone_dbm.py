"""Zone/octagon-lite abstract domain via difference-bound matrices (SYNTHETIC).

A zone over vars x1..xn is a conjunction of constraints x_i - x_j <= c,
represented as a DBM closed under shortest paths (Floyd-Warshall). Operations:
intersection (min), constraint tightening, bound extraction x<=c / x>=c.
Emptiness = negative diagonal entry. Classic timed-automata/AI domain.
"""

from __future__ import annotations

_SEED = 20261231 + 1007

INF = float("inf")


class Zone:
    """DBM over n vars + origin 0. D[i][j] = bound on xi - xj <= D[i][j]."""

    def __init__(self, n: int):
        self.n = n
        self.d = [[0.0 if i == j else INF for j in range(n + 1)] for i in range(n + 1)]

    def copy(self) -> Zone:
        z = Zone(self.n)
        z.d = [row[:] for row in self.d]
        return z

    def close(self) -> None:
        """Tighten by shortest paths (canonical form)."""
        n = self.n + 1
        d = self.d
        for k in range(n):
            for i in range(n):
                if d[i][k] == INF:
                    continue
                for j in range(n):
                    via = d[i][k] + d[k][j]
                    if via < d[i][j]:
                        d[i][j] = via

    def add_diff(self, i: int, j: int, c: float) -> None:
        """xi - xj <= c"""
        if c < self.d[i][j]:
            self.d[i][j] = c
            self.close()

    def add_upper(self, i: int, c: float) -> None:
        """xi <= c  (xi - x0 <= c)"""
        self.add_diff(i, 0, c)

    def add_lower(self, i: int, c: float) -> None:
        """xi >= c  (x0 - xi <= -c)"""
        self.add_diff(0, i, -c)

    def bound(self, i: int) -> tuple[float, float]:
        """(lo, hi) for xi."""
        hi = self.d[i][0]
        lo = -self.d[0][i]
        return (lo, hi)

    def diff_bound(self, i: int, j: int) -> float:
        return self.d[i][j]

    def is_empty(self) -> bool:
        return any(self.d[i][i] < 0 for i in range(self.n + 1))

    def intersect(self, other: Zone) -> Zone:
        z = Zone(self.n)
        z.d = [
            [min(self.d[i][j], other.d[i][j]) for j in range(self.n + 1)] for i in range(self.n + 1)
        ]
        z.close()
        return z


def bench_zone_dbm(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # x1<=5, x2<=3, x1-x2<=1 => x1<=min(5,3+1)=4 discovered by closure
    z = Zone(2)
    z.add_upper(1, 5.0)
    z.add_upper(2, 3.0)
    z.add_diff(1, 2, 1.0)
    checks.append(z.bound(1) == (-INF, 4.0))
    checks.append(z.diff_bound(1, 2) == 1.0)
    # transitivity: x1-x2<=1, x2-x3<=2 => x1-x3<=3
    z2 = Zone(3)
    z2.add_diff(1, 2, 1.0)
    z2.add_diff(2, 3, 2.0)
    checks.append(z2.diff_bound(1, 3) <= 3.0 + 1e-12)
    # emptiness: x<=2 and x>=5
    z3 = Zone(1)
    z3.add_upper(1, 2.0)
    z3.add_lower(1, 5.0)
    checks.append(z3.is_empty())
    # intersection keeps tightest
    z4 = Zone(1)
    z4.add_upper(1, 7.0)
    z5 = Zone(1)
    z5.add_upper(1, 6.0)
    z6 = z4.intersect(z5)
    checks.append(z6.bound(1) == (-INF, 6.0))
    # lower propagation: y<=x+2 => y-x<=2 with x<=4 => y<=6
    z7 = Zone(2)
    z7.add_diff(2, 1, 2.0)
    z7.add_upper(1, 4.0)
    checks.append(z7.bound(2)[1] <= 6.0 + 1e-12)
    return {"synthetic_zone_dbm": float(sum(checks)) / len(checks)}
