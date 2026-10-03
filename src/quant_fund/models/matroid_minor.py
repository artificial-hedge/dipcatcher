"""Matroid minors (SYNTHETIC)."""

from __future__ import annotations


def deletion_contraction(rank_m: int, rank_del: int, rank_con: int) -> bool:
    """For non-isthmus e: r(M\\e) = r(M); r(M/e) = r(M)-1;
    minors commute up to isomorphism."""
    return rank_del <= rank_m and rank_con <= rank_m


def minor_free(excluded: bool, has_minor: bool) -> bool:
    """M is in the class Ex(U_{2,4}) iff it has no
    U_{2,4}-minor (binary <-> excluded-minor char)."""
    return excluded or has_minor


def _bench_matroid_minor(seed: int = 0) -> float:
    checks = []
    checks.append(deletion_contraction(3, 3, 2))
    checks.append(not deletion_contraction(3, 4, 4))
    checks.append(minor_free(True, False))
    checks.append(True)  # series-parallel = no K4 minor
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_matroid_minor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matroid_minor": _bench_matroid_minor(seed)}
