"""Homotopy levels on finite types.

n=-2: contractible (exactly one element + all equal);
n=-1: mere proposition (any two elements equal);
n=0 : set (any two parallel paths equal — UIP on the discrete model);
n>=1: higher structure not collapsible.
Checkers enumerate the finite carrier exhaustively.
"""

from __future__ import annotations

_SEED = 20261231 + 1054


def is_contr(carrier: list) -> bool:
    return len(set(carrier)) == 1


def is_prop(carrier: list, eq=lambda a, b: a == b) -> bool:
    return all(eq(a, b) for a in carrier for b in carrier)


def is_set(carrier: list, paths: dict | None = None) -> bool:
    """Discrete finite type is a set when the only paths are refl.
    `paths` optionally gives distinct parallel paths between points."""
    if not paths:
        return True  # discrete finite carrier: identity is decidable
    # if any two distinct paths coexist between same endpoints -> not a set
    return all(len(ps) <= 1 for (_x, _y), ps in paths.items())


def hlevel(carrier: list, paths: dict | None = None) -> int:
    if is_contr(carrier):
        return -2
    if is_prop(carrier):
        return -1
    if is_set(carrier, paths):
        return 0
    return 1


def truncation_level(carrier: list) -> int:
    """h-level after propositional truncation: always a prop (-1)."""
    return -1 if carrier else -1


def bench_hlevel_check(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    checks.append(is_contr([7]) and hlevel([7]) == -2)
    # prop: carrier where all elements declared equal (e.g. unit-like)
    checks.append(is_prop(["u"], eq=lambda _a, _b: True))
    # singleton set of distinct values: prop iff all equal; [1,2] not prop
    checks.append(not is_prop([1, 2]))
    # bool is a set, not a prop
    checks.append(hlevel([True, False]) == 0)
    # two parallel paths between same endpoints -> not a set
    checks.append(not is_set([0, 1], paths={(0, 1): ["p", "q"]}))
    # truncating any inhabited type gives a proposition
    checks.append(truncation_level([1, 2, 3]) == -1)
    return {"synthetic_hlevel_check": float(sum(checks)) / len(checks)}
