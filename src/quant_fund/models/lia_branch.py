"""Integer linear arithmetic: bounded ILP via LP-relaxation + branch-and-bound (SYNTHETIC bench)."""

from __future__ import annotations

INF = float("inf")


def _lp_feasible(
    constraints: list[tuple[list[float], float]], bounds: list[tuple[float, float]], n: int
) -> bool:
    """Fourier–Motzkin-style feasibility over reals, variable by variable."""
    cons = list(constraints)
    for i in range(n):
        lo, hi = bounds[i]
        cons.append(([1.0 if j == i else 0.0 for j in range(n)], hi))  # x_i <= hi
        cons.append(([-1.0 if j == i else 0.0 for j in range(n)], -lo))  # -x_i <= -lo
    # eliminate variables n-1..0
    for v in range(n - 1, -1, -1):
        pos = [c for c in cons if c[0][v] > 1e-12]
        neg = [c for c in cons if c[0][v] < -1e-12]
        zero = [c for c in cons if abs(c[0][v]) <= 1e-12]
        cons = zero
        for a, b1 in pos:
            for a2, b2 in neg:
                av, bv = a[v], -a2[v]
                new = [av * a2[j] + bv * a[j] for j in range(n)]
                cons.append((new, av * b2 + bv * b1))
        for a, b in cons:
            if all(abs(c) < 1e-12 for c in a) and b < -1e-9:
                return False
    return all(not (all(abs(c) < 1e-12 for c in a) and b < -1e-9) for a, b in cons)


def ilp_solve(
    constraints: list[tuple[list[float], float]],
    n: int,
    lo_hi: tuple[int, int] = (0, 9),
    depth: int = 40,
) -> list[int] | None:
    """Find integer x in [lo,hi]^n with sum a_j x_j <= b for every (a,b)."""
    bounds: list[tuple[float, float]] = [(float(lo_hi[0]), float(lo_hi[1]))] * n
    return _bb(constraints, bounds, n, depth)


def _bb(
    constraints: list[tuple[list[float], float]],
    bounds: list[tuple[float, float]],
    n: int,
    depth: int,
) -> list[int] | None:
    if depth <= 0 or not _lp_feasible(constraints, bounds, n):
        return None
    # integral check: all bounds singletons
    if all(lo == hi for lo, hi in bounds):
        xs = [int(lo) for lo, _ in bounds]
        if all(sum(a[j] * xs[j] for j in range(n)) <= b + 1e-9 for a, b in constraints):
            return xs
        return None
    # branch on widest bound
    i = max(range(n), key=lambda j: bounds[j][1] - bounds[j][0])
    lo, hi = bounds[i]
    mid = int(lo + (hi - lo) // 2)
    for split in ((lo, mid), (mid + 1, hi)):
        if split[0] > split[1]:
            continue
        b2 = list(bounds)
        b2[i] = (float(split[0]), float(split[1]))
        out = _bb(constraints, b2, n, depth - 1)
        if out is not None:
            return out
    return None


def _bench_lia_branch(seed: int = 0) -> float:
    del seed
    checks = []
    # x + y <= 5, x - y >= 1 (i.e. -x + y <= -1) -> e.g. (3,1)
    xs = ilp_solve([([1.0, 1.0], 5.0), ([-1.0, 1.0], -1.0)], 2)
    checks.append(xs is not None and xs[0] + xs[1] <= 5 and xs[0] - xs[1] >= 1)
    # unsat: x <= 1 and x >= 2
    checks.append(ilp_solve([([1.0], 1.0), ([-1.0], -2.0)], 1) is None)
    # integer-ness matters: x*2 <= 3 has real sol 1.5 but int sol x<=1; plus -x<=-1
    xs2 = ilp_solve([([2.0], 3.0), ([-1.0], -1.0)], 1)
    checks.append(xs2 == [1])
    # LP-feasible but ILP-infeasible: 2x+2y<=5, x+y>=2.6 -> -x-y<=-2.6
    checks.append(ilp_solve([([2.0, 2.0], 5.0), ([-1.0, -1.0], -2.6)], 2) is None)
    # three vars
    xs3 = ilp_solve([([1.0, 1.0, 1.0], 4.0), ([-1.0, 0.0, 0.0], -1.0)], 3)
    checks.append(xs3 is not None and sum(xs3) <= 4 and xs3[0] >= 1)
    return sum(checks) / len(checks)


def bench_lia_branch(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lia_branch": _bench_lia_branch(seed)}
