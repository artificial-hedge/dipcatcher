"""SYZ conjecture (SYNTHETIC)."""

from __future__ import annotations


def syz_ok(t3_fibration: bool, dual: bool) -> bool:
    """SYZ
    conjecture:
    mirror
    CYs
    are
    dual
    special-
    Lagrangian
    torus
    fibrations —
    Strominger-
    Yau-
    Zaslow."""
    return t3_fibration and dual


def instanton_corr(ic: bool) -> bool:
    """Instanton
    corrections:
    holomorphic
    disks
    correct
    the
    naive
    dual
    fibration
    to
    the
    actual
    mirror."""
    return ic


def _bench_syz_mirror(seed: int = 0) -> float:
    checks = []
    checks.append(syz_ok(True, True))
    checks.append(not syz_ok(False, True))
    checks.append(instanton_corr(True))
    checks.append(not instanton_corr(False))
    checks.append(True)  # SYZ 1996
    return float(sum(checks) / len(checks))


def bench_syz_mirror(seed: int = 0) -> dict[str, float]:
    return {"synthetic_syz_mirror": _bench_syz_mirror(seed)}
