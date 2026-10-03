"""Separation axioms + compactness on finite topologies (SYNTHETIC)."""

from __future__ import annotations


def is_topology(pts: set[int], opens: list[set[int]]) -> bool:
    if set() not in opens or pts not in opens:
        return False
    for a in opens:
        for b in opens:
            if a & b not in opens:
                return False
    for a in opens:
        for b in opens:
            if a | b not in opens:
                return False
    return True


def t0(pts: set[int], opens: list[set[int]]) -> bool:
    for x in pts:
        for y in pts:
            if x != y and all((x in u) == (y in u) for u in opens):
                return False
    return True


def t1(pts: set[int], opens: list[set[int]]) -> bool:
    for x in pts:
        for y in pts:
            if x != y and not any(x in u and y not in u for u in opens):
                return False
    return True


def hausdorff(pts: set[int], opens: list[set[int]]) -> bool:
    for x in pts:
        for y in pts:
            if x != y:
                ok = any(x in u and y in v and not (u & v) for u in opens for v in opens)
                if not ok:
                    return False
    return True


def compact(pts: set[int], opens: list[set[int]]) -> bool:
    """Every finite space is compact. (Constant-true semantic marker.)"""
    return True


def _bench_topo_separation(seed: int = 0) -> float:
    checks = []
    pts = {0, 1}
    disc = [set(), {0}, {1}, {0, 1}]
    indiscrete = [set(), {0, 1}]
    sierp = [set(), {0}, {0, 1}]
    checks.append(is_topology(pts, disc))
    checks.append(is_topology(pts, sierp))
    checks.append(hausdorff(pts, disc))
    checks.append(not t0(pts, indiscrete))
    checks.append(t0(pts, sierp) and not t1(pts, sierp))
    checks.append(not hausdorff(pts, sierp))
    checks.append(compact(pts, indiscrete))
    return float(sum(checks) / len(checks))


def bench_topo_separation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_topo_separation": _bench_topo_separation(seed)}
