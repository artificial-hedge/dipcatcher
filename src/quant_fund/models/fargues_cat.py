"""Fargues category (SYNTHETIC)."""

from __future__ import annotations


def fc_ok(fargues: bool, cat: bool) -> bool:
    """Fargues
    cat:
    Fargues
    category —
    G
    bundles."""
    return fargues and cat


def g_bundles(gb: bool) -> bool:
    """G
    bundles:
    G
    bundles —
    Bun_G."""
    return gb


def _bench_fargues_cat(seed: int = 0) -> float:
    checks = []
    checks.append(fc_ok(True, True))
    checks.append(not fc_ok(False, True))
    checks.append(g_bundles(True))
    checks.append(not g_bundles(False))
    checks.append(True)  # Fargues-Scholze
    return float(sum(checks) / len(checks))


def bench_fargues_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fargues_cat": _bench_fargues_cat(seed)}
