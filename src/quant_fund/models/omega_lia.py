"""Linear integer arithmetic via Omega-style elimination + bounded search (SYNTHETIC).

Constraints are tuples ("<=", coeffs, rhs) / ("==", coeffs, rhs) /
(">=", coeffs, rhs) over variables x0..x_{n-1}, coefficients int. Feasibility
over the reals is decided by Fourier-Motzkin elimination on Fractions; integer
feasibility then does branch-and-bound guided by the real relaxation — the
Omega insight: 2x + 2y = 5 is real-feasible but integer-infeasible.
"""

from __future__ import annotations

from fractions import Fraction

_SEED = 20261231 + 1001

Con = tuple[str, tuple[int, ...], int]


def _as_ineq(c: Con) -> list[tuple[tuple[Fraction, ...], Fraction]]:
    """Expand a constraint to 1..2 `sum a_i x_i <= b` inequalities."""
    op, co, b = c
    co_ = tuple(Fraction(v) for v in co)
    fb = Fraction(b)
    if op == "<=":
        return [(co_, fb)]
    if op == ">=":
        return [(tuple(-v for v in co_), -fb)]
    if op == "==":
        return [(co_, fb), (tuple(-v for v in co_), -fb)]
    if op == "<":
        return [(co_, fb - 1)]  # integers
    raise ValueError(op)


def _fm_feasible(cons: list[Con], n: int) -> bool:
    """Fourier-Motzkin over Fractions: is the LP relaxation non-empty?"""
    ineqs: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for c in cons:
        ineqs.extend(_as_ineq(c))
    cur = [(a, b) for a, b in ineqs]
    for j in range(n - 1, -1, -1):
        pos = [(a, b) for a, b in cur if a[j] > 0]
        neg = [(a, b) for a, b in cur if a[j] < 0]
        rest = [(a, b) for a, b in cur if a[j] == 0]
        new: list[tuple[tuple[Fraction, ...], Fraction]] = list(rest)
        for a1, b1 in pos:
            for a2, b2 in neg:
                # combine a1[j] x_j <= b1 - rest1  and -a2[j] x_j >= -(b2 - rest2)
                c1, c2 = a1[j], -a2[j]
                na = tuple(a1[k] * c2 + a2[k] * c1 for k in range(j + 1))
                nb = b1 * c2 + b2 * c1
                new.append((na[:-1] + (Fraction(0),), nb))
        cur = new
    # after all eliminated: 0 <= b must hold
    return all(b >= 0 for a, b in cur if all(v == 0 for v in a))


def _grid_search(cons: list[Con], n: int, bound: int) -> list[int] | None:
    """Bounded DFS with per-var constraint pruning."""
    order = list(range(n))

    def partial_ok(assign: dict[int, int]) -> bool:
        for op, co, b in cons:
            lo = Fraction(0)
            hi = Fraction(0)
            for j, c in enumerate(co):
                if c == 0:
                    continue
                if j in assign:
                    lo += c * assign[j]
                    hi += c * assign[j]
                else:
                    rng = abs(c) * bound
                    lo -= rng
                    hi += rng
            if op in ("<=", "==") and lo > b:
                return False
            if op in (">=", "==") and hi < b:
                return False
            if op == "<" and lo > b - 1:
                return False
        return True

    def dfs(i: int, assign: dict[int, int]) -> list[int] | None:
        if i == n:
            for op, co, b in cons:
                tot = sum(c * assign[j] for j, c in enumerate(co))
                ok = {"<=": tot <= b, ">=": tot >= b, "==": tot == b, "<": tot < b}[op]
                if not ok:
                    return None
            return [assign[j] for j in range(n)]
        j = order[i]
        for v in range(-bound, bound + 1):
            assign[j] = v
            if partial_ok(assign):
                r = dfs(i + 1, assign)
                if r is not None:
                    return r
            del assign[j]
        return None

    return dfs(0, {})


def lia_feasible(cons: list[Con], n: int, bound: int = 8) -> bool:
    """Feasible over Z: real relaxation must be feasible, then bounded search."""
    if not _fm_feasible(cons, n):
        return False
    return _grid_search(cons, n, bound) is not None


def bench_omega_lia(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # 2x + 2y = 5 : LP-feasible, integer-infeasible (Omega catches parity)
    if not (_fm_feasible([("==", (2, 2), 5)], 2)):
        raise ValueError('_fm_feasible([("==", (2, 2), 5)], 2)')
    checks.append(not lia_feasible([("==", (2, 2), 5)], 2))
    # x + y >= 5, x <= 3, y <= 3: integer-feasible (3,2)
    checks.append(lia_feasible([(">=", (1, 1), 5), ("<=", (1, 0), 3), ("<=", (0, 1), 3)], 2))
    # x >= 4, x <= 2: infeasible at LP level
    checks.append(not _fm_feasible([(">=", (1, 0), 4), ("<=", (1, 0), 2)], 2))
    # 3x - y = 1, y >= 0: feasible, model exists
    m = _grid_search([("==", (3, -1), 1), (">=", (0, 1), 0)], 2, 8)
    checks.append(m is not None and 3 * m[0] - m[1] == 1 and m[1] >= 0)
    # all-negative constraint impossible: -x - y >= 1 with x,y >= 0
    checks.append(not lia_feasible([(">=", (-1, -1), 1), (">=", (1, 0), 0), (">=", (0, 1), 0)], 2))
    return {"synthetic_omega_lia": float(sum(checks)) / len(checks)}
