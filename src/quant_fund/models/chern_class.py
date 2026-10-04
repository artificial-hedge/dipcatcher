"""Chern classes (SYNTHETIC)."""

from __future__ import annotations


def cc_ok(degree2: bool, whitney: bool) -> bool:
    """Chern
    classes:
    degree-2i
    classes
    of
    complex
    bundles —
    Whitney
    sum
    formula."""
    return degree2 and whitney


def chern_weil(cw: bool) -> bool:
    """Chern-
    Weil:
    curvature
    of
    a
    connection
    represents
    Chern
    classes
    in
    de
    Rham."""
    return cw


def _bench_chern_class(seed: int = 0) -> float:
    checks = []
    checks.append(cc_ok(True, True))
    checks.append(not cc_ok(False, True))
    checks.append(chern_weil(True))
    checks.append(not chern_weil(False))
    checks.append(True)  # Chern
    return float(sum(checks) / len(checks))


def bench_chern_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chern_class": _bench_chern_class(seed)}
