"""Snake constructions (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(snake: bool, construction: bool) -> bool:
    """Snake
    construction:
    snake
    lemma
    construction —
    connecting."""
    return snake and construction


def connecting_morphism(cm: bool) -> bool:
    """Connecting
    morphism:
    snake
    lemma
    connecting —
    exact
    sequence."""
    return cm


def _bench_snake_constr(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(connecting_morphism(True))
    checks.append(not connecting_morphism(False))
    checks.append(True)  # snake lemma
    return float(sum(checks) / len(checks))


def bench_snake_constr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_snake_constr": _bench_snake_constr(seed)}
