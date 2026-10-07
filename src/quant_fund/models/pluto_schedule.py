"""Pluto-lite affine scheduler for loop nests (SYNTHETIC).

Statements iterate over index vectors; dependences are distance vectors
dst - src (constant). A schedule is per-statement coefficients s_i where
execution time = s·i + c. Legal iff for every dep the scheduled dst time
>= src time, componentwise-lexicographic positive when carrying.
We enumerate small integer coefficient candidates (the SOTA solvers use
Farkas LP; enumeration is exact over small coefficient space) and pick
the schedule maximizing outer-permutable loops (all distances >=0).
"""

from __future__ import annotations

from itertools import product

_SEED = 20261231 + 1031

Dep = tuple[int, ...]


def _legal(schedule: tuple[int, ...], deps: list[Dep]) -> bool:
    """All dep distances mapped by schedule are >=0 (componentwise >0 at
    first nonzero — we require s·d >= 0 for lex-positivity)."""
    return all(sum(s * d for s, d in zip(schedule, dep, strict=True)) >= 0 for dep in deps)


def _first_legal_component(schedule: tuple[int, ...], deps: list[Dep]) -> bool:
    return all(sum(s * d for s, d in zip(schedule, dep, strict=True)) >= 0 for dep in deps)


def schedule(deps: list[Dep], ndims: int, max_coeff: int = 2) -> tuple[int, ...] | None:
    """Return a legal coefficient schedule minimizing reuse damage —
    prefer schedules where more deps get distance 0 (parallelizable
    time carried by outer loops)."""
    best: tuple[tuple[int, int], tuple[int, ...]] | None = None
    for cand in product(range(max_coeff + 1), repeat=ndims):
        if all(c == 0 for c in cand):
            continue
        if not _legal(cand, deps):
            continue
        zeros = sum(1 for dep in deps if sum(s * d for s, d in zip(cand, dep, strict=True)) == 0)
        norm = sum(cand)
        key = (-zeros, norm)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1] if best else None


def permutable(deps: list[Dep]) -> bool:
    """Nest fully permutable (tileable in all dims) iff every dep vector is
    componentwise >= 0."""
    return all(all(d >= 0 for d in dep) for dep in deps)


def bench_pluto_schedule(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # wavefront dep (1,-1): schedule must satisfy s1 - s2 >= 0
    s = schedule([(1, -1)], 2, max_coeff=3)
    checks.append(s is not None and s[0] - s[1] >= 0)
    # all-positive deps: identity schedule legal
    s2 = schedule([(1, 0), (0, 1)], 2)
    checks.append(s2 is not None and _legal(s2, [(1, 0), (0, 1)]))
    # permutable nest -> true
    checks.append(permutable([(1, 0), (0, 1)]))
    # negative component -> not permutable
    checks.append(not permutable([(1, -1)]))
    # infeasible: dep (-1,0) in a nest needs s*(-1)>=0 => s=0 only (rejected as all-zero)
    checks.append(schedule([(-1,)], 1, max_coeff=2) is None)
    # 1D dep (1,): any positive schedule works; picks smallest coeff
    checks.append(schedule([(1,)], 1) == (1,))
    return {"synthetic_pluto_schedule": float(sum(checks)) / len(checks)}
