"""Quantifier elimination for dense linear orders (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction

# Atoms over variable x and parameters y_i: ("lt", "x", i) means x < y_i,
# ("gt", i, "x") means y_i < x, ("eq", "x", i) means x = y_i.
Atom = tuple[str, str | int, str | int]


def eliminate_exists(atoms: list[Atom], n_params: int) -> list[tuple[int, int, int]]:
    """Exists x. AND(atoms) over a dense order without endpoints.

    Returns a list of comparison facts among parameters implied by the
    existential: tuples (kind, i, j) with kind 0: y_i < y_j, 1: y_i = y_j,
    2: y_i <= y_j — the qf formula is the conjunction of required ones.
    We encode the result as the set of *consistent orientations*: to stay
    executable we return the strongest implied comparisons: for every pair
    (lower bound y_i, upper bound y_j) require y_i < y_j unless an equality
    x = y_k witnesses it.
    """
    eqs = [a[2] for a in atoms if a[0] == "eq"]
    lowers = [a[1] for a in atoms if a[0] == "gt"]  # y_i < x
    uppers = [a[2] for a in atoms if a[0] == "lt"]  # x < y_j
    out: list[tuple[int, int, int]] = []
    if eqs:
        k = int(eqs[0])
        # x = y_k satisfies all strict bounds iff y_i < y_k < y_j
        for i in lowers:
            out.append((0, int(i), k))
        for j in uppers:
            out.append((0, k, int(j)))
        for e in eqs[1:]:
            out.append((1, k, int(e)))
        return out
    for i in lowers:
        for j in uppers:
            out.append((0, int(i), int(j)))
    return out


def eval_qf(comps: list[tuple[int, int, int]], y: list[Fraction]) -> bool:
    return all(
        (kind == 0 and y[i] < y[j]) or (kind == 1 and y[i] == y[j]) or (kind == 2 and y[i] <= y[j])
        for kind, i, j in comps
    )


def brute_exists(atoms: list[Atom], y: list[Fraction], grid: list[Fraction]) -> bool:
    for x in grid:
        ok = True
        for a in atoms:
            if (
                (a[0] == "lt" and x >= y[int(a[2])])
                or (a[0] == "gt" and y[int(a[1])] >= x)
                or (a[0] == "eq" and x != y[int(a[2])])
            ):
                ok = False
            if not ok:
                break
        if ok:
            return True
    return False


def _bench_quantifier_elim(seed: int = 0) -> float:
    checks = []
    grid = [Fraction(k, 7) for k in range(-14, 15)]  # dense-enough rationals
    # exists x: y0 < x < y1  iff  y0 < y1
    atoms: list[Atom] = [("gt", 0, "x"), ("lt", "x", 1)]
    comps = eliminate_exists(atoms, 2)
    for y0 in [Fraction(-1), Fraction(0), Fraction(1)]:
        for y1 in [Fraction(-1), Fraction(0), Fraction(1)]:
            checks.append(eval_qf(comps, [y0, y1]) == brute_exists(atoms, [y0, y1], grid))
    # exists x: x = y0 and x < y1 iff y0 < y1
    atoms2: list[Atom] = [("eq", "x", 0), ("lt", "x", 1)]
    comps2 = eliminate_exists(atoms2, 2)
    for y0 in [Fraction(-2), Fraction(0)]:
        for y1 in [Fraction(-1), Fraction(1)]:
            checks.append(eval_qf(comps2, [y0, y1]) == brute_exists(atoms2, [y0, y1], grid))
    # exists x: y0 < x, x < y1, y2 < x -> y0<y1 and y2<y1
    atoms3: list[Atom] = [("gt", 0, "x"), ("lt", "x", 1), ("gt", 2, "x")]
    comps3 = eliminate_exists(atoms3, 3)
    for y0 in [Fraction(-1), Fraction(1)]:
        for y1 in [Fraction(0), Fraction(2)]:
            for y2 in [Fraction(-1), Fraction(3)]:
                y = [y0, y1, y2]
                checks.append(eval_qf(comps3, y) == brute_exists(atoms3, y, grid))
    return float(sum(checks) / len(checks))


def bench_quantifier_elim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quantifier_elim": _bench_quantifier_elim(seed)}
