"""MCSAT-lite: model-constructing SAT over integer arithmetic (SYNTHETIC bench)."""

from __future__ import annotations

# clauses: disjunctions of literals; a literal is (var, lo, hi) meaning lo <= var <= hi
# (negations expressed by complementing the bound pair outside this api)


def _lit_ok(assign: dict[str, int], lit: tuple) -> bool:
    v, lo, hi = lit
    if v not in assign:
        return False  # unassigned -> not satisfied yet
    return bool(lo <= assign[v] <= hi)


def _lit_conflict(assign: dict[str, int], lit: tuple) -> bool:
    v, lo, hi = lit
    return v in assign and not (lo <= assign[v] <= hi)


def mcsat_solve(
    clauses: list[list[tuple]],
    vars_: list[str],
    dom: tuple[int, int] = (-9, 9),
    iters: int = 2000,
) -> dict[str, int] | None:
    """Greedy model construction with conflict-driven value changes.

    Maintain an assignment; repeatedly pick a violated clause, change one of
    its vars to a satisfying value inside the domain; if no value works for
    any var (conflict), backtrack — approximated by restart with a different
    seed choice. Good enough for small toy benches.
    """
    import random

    rng = random.Random(20261231 + 1065)
    assign: dict[str, int] = {}
    steps = 0
    while steps < iters:
        steps += 1
        bad = [c for c in clauses if _violated(c, assign)]
        if not bad:
            if all(v in assign for v in vars_):
                return assign
            v = next(v for v in vars_ if v not in assign)
            # pick value satisfying as many clauses containing v as possible
            best, bestv = -1, assign.get(v, 0)
            for val in range(dom[0], dom[1] + 1):
                cand = {**assign, v: val}
                score = sum(0 if _violated(c, cand) else 1 for c in clauses)
                if score > best or (score == best and rng.random() < 0.3):
                    best, bestv = score, val
            assign[v] = bestv
            continue
        c = bad[rng.randrange(len(bad))]
        # try each literal of clause, each value, pick the one minimizing violations
        best, move = -(len(clauses) + 1), None
        for lit in c:
            v = lit[0]
            for val in range(dom[0], dom[1] + 1):
                cand = {**assign, v: val}
                if _violated(c, cand):
                    continue
                score = -sum(1 for c2 in clauses if _violated(c2, cand))
                if score > best:
                    best, move = score, (v, val)
        if move is None:
            return None  # clause unsatisfiable under any assignment to its vars
        assign[move[0]] = move[1]
    return assign if not any(_violated(c, assign) for c in clauses) else None


def _violated(clause: list[tuple], assign: dict[str, int]) -> bool:
    """Clause violated iff every literal is assigned and false."""
    return all(lit[0] in assign and not _lit_ok(assign, lit) for lit in clause)


def _bench_mcsat_lite(seed: int = 0) -> float:
    del seed
    checks = []
    sol = mcsat_solve([[("x", 0, 3)], [("x", -9, 1)]], ["x"])
    checks.append(sol is not None and 0 <= sol["x"] <= 1)
    sol2 = mcsat_solve([[("x", 0, 5), ("y", 0, 5)], [("x", 2, 9)], [("y", -9, -1)]], ["x", "y"])
    # x in [2,5] and y in [-9,-1] works; clause1 forces x<=5 or y<=5 (both ok)
    checks.append(sol2 is not None and 2 <= sol2["x"] <= 5 and sol2["y"] <= -1)
    # UNSAT: x<=1 and x>=2
    checks.append(mcsat_solve([[("x", -9, 1)], [("x", 2, 9)]], ["x"]) is None)
    # disjunction escape
    sol4 = mcsat_solve([[("x", 0, 1), ("x", 4, 5)], [("x", 1, 4)]], ["x"])
    checks.append(sol4 is not None and sol4["x"] in (1, 4))
    return sum(checks) / len(checks)


def bench_mcsat_lite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mcsat_lite": _bench_mcsat_lite(seed)}
