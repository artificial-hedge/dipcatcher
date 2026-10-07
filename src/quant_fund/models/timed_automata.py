"""Timed automaton reachability via zone exploration (SYNTHETIC).

Zones are DBMs over clocks {x0=0,x1,...,xn}: (lo,hi) bounds on xi-xj.
Operations: up (unbounded elapse), reset, intersect-guard (clock cmp
const). Forward zone graph BFS decides whether a target location is
reachable — the classic UPPAAL approach.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1041

INF = float("inf")
# zone: (lo,hi) per clock vs reference 0: lo <= x <= hi; guards are (var,"<=|>=",c)
Zone = list[tuple[float, float]]


def up(z: Zone) -> Zone:
    return [(lo, INF) for lo, hi in z]


def reset(z: Zone, var: int) -> Zone:
    out = list(z)
    out[var] = (0.0, 0.0)
    return out


def guard(z: Zone, var: int, cmp_: str, c: float) -> Zone | None:
    lo, hi = z[var]
    if cmp_ == "<=":
        hi = min(hi, c)
    elif cmp_ == ">=":
        lo = max(lo, c)
    elif cmp_ == "<":
        hi = min(hi, c - 1e-9)
    elif cmp_ == ">":
        lo = max(lo, c + 1e-9)
    elif cmp_ == "==":
        lo = max(lo, c)
        hi = min(hi, c)
    if lo > hi + 1e-9:
        return None
    z2 = list(z)
    z2[var] = (lo, hi)
    return z2


def _empty(z: Zone) -> bool:
    return any(lo > hi + 1e-9 for lo, hi in z)


def reachable(
    locs: dict[str, dict[str, Any]],
    edges: list[tuple],
    start: tuple[str, Zone],
    target: str,
    bound: int = 60,
) -> bool:
    """edges: (src,dst,guards,resets) — guards list of (var,cmp,c)."""
    seen: set[tuple[str, tuple[tuple[float, float], ...]]] = set()
    wl = [start]
    steps = 0
    while wl and steps < bound * 100:
        steps += 1
        loc, z = wl.pop()
        key = (loc, tuple((round(lo, 6), round(hi, 6)) for lo, hi in z))
        if key in seen:
            continue
        seen.add(key)
        if loc == target:
            return True
        for src, dst, gs, rs in edges:
            if src != loc:
                continue
            z2: Zone | None = up(list(z))
            ok = True
            for v, c_, k in gs:
                if z2 is None:
                    ok = False
                    break
                z2 = guard(z2, v, c_, k)
                if z2 is None:
                    ok = False
                    break
            if not ok or z2 is None:
                continue
            for v in rs:
                z2 = reset(z2, v)
            if not _empty(z2):
                wl.append((dst, z2))
    return False


def bench_timed_automata(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # l0 --[x<=2, reset x]--> l1 --[x>=1]--> l2
    edges = [
        ("l0", "l1", [(0, "<=", 2.0)], [0]),
        ("l1", "l2", [(0, ">=", 1.0)], []),
    ]
    z0: Zone = [(0.0, 0.0)]
    checks.append(reachable({}, edges, ("l0", z0), "l2"))
    # l0 --[x<=2]--> l1 --[x>=5]--> l2: can't wait past x<=2 then x>=5? time
    # elapses in l1 freely — l1 has no invariant so x can exceed 5 -> reachable
    edges2: list[tuple] = [
        ("l0", "l1", [(0, "<=", 2.0)], []),
        ("l1", "l2", [(0, ">=", 5.0)], []),
    ]
    checks.append(reachable({}, edges2, ("l0", z0), "l2"))
    # l0 --[x<=2, reset x]--> l1 --[x>=5]--> l2: after reset, wait 5s -> reachable
    edges3 = [
        ("l0", "l1", [(0, "<=", 2.0)], [0]),
        ("l1", "l2", [(0, ">=", 5.0)], []),
    ]
    checks.append(reachable({}, edges3, ("l0", z0), "l2"))
    # direct contradiction guard: x<=1 and x>=3 -> unreachable
    edges4: list[tuple] = [("l0", "l2", [(0, "<=", 1.0), (0, ">=", 3.0)], [])]
    checks.append(not reachable({}, edges4, ("l0", z0), "l2"))
    # guard zone math
    z0b: Zone = [(0.0, INF)]
    z = guard(z0b, 0, "<=", 2.0)
    if not (z is not None):
        raise ValueError("z is not None")
    z = reset(z, 0)
    z = up(z)
    z = guard(z, 0, ">=", 1.0)
    checks.append(z is not None and z[0][0] == 1.0)
    return {"synthetic_timed_automata": float(sum(checks)) / len(checks)}
