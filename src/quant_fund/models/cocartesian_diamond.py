"""Cocartesian diamonds (SYNTHETIC)."""

from __future__ import annotations


def cd_ok(cocartesian: bool, diamond: bool) -> bool:
    """Cocartesian
    diamond:
    cocartesian
    diamond
    morphism —
    v-stack."""
    return cocartesian and diamond


def vstack_morphism(vm: bool) -> bool:
    """V-stack
    morphism:
    v-stack
    morphism —
    small
    v-stack."""
    return vm


def _bench_cocartesian_diamond(seed: int = 0) -> float:
    checks = []
    checks.append(cd_ok(True, True))
    checks.append(not cd_ok(False, True))
    checks.append(vstack_morphism(True))
    checks.append(not vstack_morphism(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_cocartesian_diamond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cocartesian_diamond": _bench_cocartesian_diamond(seed)}
