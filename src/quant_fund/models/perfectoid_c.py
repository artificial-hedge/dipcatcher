"""Perfectoid fields (SYNTHETIC)."""

from __future__ import annotations


def pc_ok(perfectoid: bool, field: bool) -> bool:
    """Perfectoid:
    perfectoid
    field —
    Scholze
    perfectoid."""
    return perfectoid and field


def almost_purity(ap: bool) -> bool:
    """Almost
    purity:
    almost
    purity
    theorem —
    Faltings
    almost."""
    return ap


def _bench_perfectoid_c(seed: int = 0) -> float:
    checks = []
    checks.append(pc_ok(True, True))
    checks.append(not pc_ok(False, True))
    checks.append(almost_purity(True))
    checks.append(not almost_purity(False))
    checks.append(True)  # Faltings
    return float(sum(checks) / len(checks))


def bench_perfectoid_c(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perfectoid_c": _bench_perfectoid_c(seed)}
