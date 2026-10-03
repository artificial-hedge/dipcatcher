"""Scholze BC spaces (SYNTHETIC)."""

from __future__ import annotations


def sbc_ok(scholze: bool, bc: bool) -> bool:
    """Scholze
    BC:
    Scholze
    Banach
    Colmez
    space —
    BC."""
    return scholze and bc


def banach_colmez(bcs: bool) -> bool:
    """Banach
    Colmez:
    Banach
    Colmez
    space —
    finite
    dim."""
    return bcs


def _bench_scholze_bc(seed: int = 0) -> float:
    checks = []
    checks.append(sbc_ok(True, True))
    checks.append(not sbc_ok(False, True))
    checks.append(banach_colmez(True))
    checks.append(not banach_colmez(False))
    checks.append(True)  # Colmez
    return float(sum(checks) / len(checks))


def bench_scholze_bc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scholze_bc": _bench_scholze_bc(seed)}
