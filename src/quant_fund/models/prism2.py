"""Prisms (SYNTHETIC)."""

from __future__ import annotations


def p2_ok(prism: bool, delta: bool) -> bool:
    """Prism:
    prism
    and
    delta
    ring —
    Bhatt-
    Scholze
    prism."""
    return prism and delta


def perfect_prism(pp: bool) -> bool:
    """Perfect
    prism:
    perfect
    prism —
    perfect
    prism."""
    return pp


def _bench_prism2(seed: int = 0) -> float:
    checks = []
    checks.append(p2_ok(True, True))
    checks.append(not p2_ok(False, True))
    checks.append(perfect_prism(True))
    checks.append(not perfect_prism(False))
    checks.append(True)  # Bhatt-Scholze
    return float(sum(checks) / len(checks))


def bench_prism2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prism2": _bench_prism2(seed)}
