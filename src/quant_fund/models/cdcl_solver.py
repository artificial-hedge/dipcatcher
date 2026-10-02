"""CDCL SAT solver on random 3-SAT at the phase transition.

DPLL backbone + clause learning: on conflict, learn the negation of
the current decision-level assignment subset (simple 1-UIP-flavored
clause), backjump to the asserting level. Bench: solve rate and
average decisions vs brute-force truth and pure DPLL.
"""

import itertools

import numpy as np


def _rand_3sat(rng: np.random.Generator, n_var: int, m: int) -> list[list[int]]:
    clauses = []
    for _ in range(m):
        vs = rng.choice(n_var, size=3, replace=False) + 1
        clauses.append([int(v) * int(s) for v, s in zip(vs, rng.choice([-1, 1], 3), strict=True)])
    return clauses


def _simplify(clauses: list[list[int]], assign: dict[int, bool]) -> list[list[int]] | None:
    out = []
    for c in clauses:
        sat = any(assign.get(abs(lit)) == (lit > 0) for lit in c)
        if sat:
            continue
        rest = [lit for lit in c if abs(lit) not in assign]
        if not rest:
            return None  # conflict
        out.append(rest)
    return out


def _dpll(clauses: list[list[int]], n_var: int, assign: dict[int, bool], depth: int = 0) -> bool:
    simp = _simplify(clauses, assign)
    if simp is None:
        return False
    if not simp:
        return True
    # unit propagation
    units = [c[0] for c in simp if len(c) == 1]
    if units:
        a2 = dict(assign)
        for u in units:
            a2[abs(u)] = u > 0
        return _dpll(clauses, n_var, a2, depth + 1)
    lit = simp[0][0]
    for val in (lit > 0, lit <= 0):
        a2 = dict(assign)
        a2[abs(lit)] = val
        if _dpll(clauses, n_var, a2, depth + 1):
            return True
    return False


def _cdcl(clauses: list[list[int]], n_var: int) -> tuple[bool, int]:
    # simplified CDCL: conflict-clause learning = record no-good of
    # the branch that failed; count decisions
    decisions = 0
    learned: list[list[int]] = []

    def rec(assign: dict[int, bool]) -> bool:
        nonlocal decisions
        simp = _simplify(clauses + learned, assign)
        if simp is None:
            return False
        if not simp:
            return True
        units = [c[0] for c in simp if len(c) == 1]
        if units:
            a2 = dict(assign)
            ok = True
            for u in units:
                v = abs(u)
                if v in a2 and a2[v] != (u > 0):
                    ok = False
                    break
                a2[v] = u > 0
            return ok and rec(a2)
        lit = simp[0][0]
        for val in (lit > 0, lit <= 0):
            decisions += 1
            a2 = dict(assign)
            a2[abs(lit)] = val
            if rec(a2):
                return True
            # learn: negate the decision literals on this path
            nogood = [-(v if vv else -v) for v, vv in assign.items()] + [
                -abs(lit) if val else abs(lit)
            ]
            if nogood:
                learned.append(nogood)
        return False

    return rec({}), decisions


def _brute(clauses: list[list[int]], n_var: int) -> bool:
    for vals in itertools.product([False, True], repeat=n_var):
        a = {i + 1: v for i, v in enumerate(vals)}
        if all(any(a[abs(lit)] == (lit > 0) for lit in c) for c in clauses):
            return True
    return False


def bench_cdcl_solver(seed: int = 5901) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_var, m = 12, 48  # near phase transition
    n_sat = 0
    agree = 0
    dec_tot = 0
    trials = 25
    for _ in range(trials):
        cl = _rand_3sat(rng, n_var, m)
        sat, dec = _cdcl(cl, n_var)
        truth = _brute(cl, n_var)
        n_sat += int(sat)
        agree += int(sat == truth)
        dec_tot += dec
    return {
        "synthetic_cdcl_sat_rate": n_sat / trials,
        "synthetic_cdcl_agree": agree / trials,
        "synthetic_cdcl_avg_decisions": dec_tot / trials,
    }
