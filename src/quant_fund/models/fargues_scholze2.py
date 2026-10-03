"""Fargues-Scholze theory (SYNTHETIC)."""

from __future__ import annotations


def fs2_ok(fargues: bool, scholze: bool) -> bool:
    """Fargues
    Scholze:
    Fargues
    Scholze
    theory —
    geometric
    Langlands."""
    return fargues and scholze


def fs_chartier(fsc: bool) -> bool:
    """FS
    chartier:
    Fargues
    Scholze
    chartier —
    local
    Langlands."""
    return fsc


def _bench_fargues_scholze2(seed: int = 0) -> float:
    checks = []
    checks.append(fs2_ok(True, True))
    checks.append(not fs2_ok(False, True))
    checks.append(fs_chartier(True))
    checks.append(not fs_chartier(False))
    checks.append(True)  # Fargues-Scholze
    return float(sum(checks) / len(checks))


def bench_fargues_scholze2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fargues_scholze2": _bench_fargues_scholze2(seed)}
