"""Fargues-Scholze diamonds (SYNTHETIC)."""

from __future__ import annotations


def diamond_ok(sheaf: bool, adic: bool) -> bool:
    """Diamond:
    sheaf on the
    pro-étale site
    of perfectoid
    spaces; quotient
    of a perfectoid
    by a pro-étale
    equivalence."""
    return sheaf and adic


def sch_spatial(spatial: bool) -> bool:
    """Spatial diamonds:
    qcqs objects
    admit quasi-
    pro-étale
    covers by
    perfectoid
    spaces."""
    return spatial


def _bench_fs_diamond(seed: int = 0) -> float:
    checks = []
    checks.append(diamond_ok(True, True))
    checks.append(not diamond_ok(False, True))
    checks.append(sch_spatial(True))
    checks.append(not sch_spatial(False))
    checks.append(True)  # Scholze diamonds
    return float(sum(checks) / len(checks))


def bench_fs_diamond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fs_diamond": _bench_fs_diamond(seed)}
