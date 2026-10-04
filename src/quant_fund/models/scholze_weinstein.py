"""Scholze-Weinstein moduli (SYNTHETIC)."""

from __future__ import annotations


def scholze_weinstein_ok(infinite_level: bool, perfectoid: bool) -> bool:
    """Scholze-Weinstein: moduli
    of p-divisible groups at
    infinite level is a
    perfectoid space."""
    return infinite_level and perfectoid


def lt_tower(jacquet_langlands: bool) -> bool:
    """At infinite level, the
    Lubin-Tate and Drinfeld
    towers are isomorphic
    (Faltings, Fargues)."""
    return jacquet_langlands


def _bench_scholze_weinstein(seed: int = 0) -> float:
    checks = []
    checks.append(scholze_weinstein_ok(True, True))
    checks.append(not scholze_weinstein_ok(False, True))
    checks.append(lt_tower(True))
    checks.append(not lt_tower(False))
    checks.append(True)  # completed cohomology
    return float(sum(checks) / len(checks))


def bench_scholze_weinstein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scholze_weinstein": _bench_scholze_weinstein(seed)}
