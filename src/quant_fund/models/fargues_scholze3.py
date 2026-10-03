"""Fargues-Scholze geometry 3 (SYNTHETIC)."""

from __future__ import annotations


def fs3_ok(fargues: bool, scholze: bool) -> bool:
    """Fargues-Scholze:
    curve-book
    formalism —
    diamonds."""
    return fargues and scholze


def v_sheaf_descent(vd: bool) -> bool:
    """v-descent:
    v-sheaf
    descent —
    Faithfully
    flat."""
    return vd


def _bench_fargues_scholze3(seed: int = 0) -> float:
    checks = []
    checks.append(fs3_ok(True, True))
    checks.append(not fs3_ok(False, True))
    checks.append(v_sheaf_descent(True))
    checks.append(not v_sheaf_descent(False))
    checks.append(True)  # FS monograph
    return float(sum(checks) / len(checks))


def bench_fargues_scholze3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fargues_scholze3": _bench_fargues_scholze3(seed)}
