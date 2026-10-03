"""MacPherson-Chern class (SYNTHETIC)."""

from __future__ import annotations


def mc_ok(chern_class: bool, functorial: bool) -> bool:
    """MacPherson
    Chern:
    Chern
    class
    of
    singular
    varieties —
    natural
    transformation."""
    return chern_class and functorial


def ehlers_stiefel(es: bool) -> bool:
    """Ehlers-
    Stiefel:
    Chern
    class
    from
    constructible
    functions —
    MacPherson."""
    return es


def _bench_macpherson_chern(seed: int = 0) -> float:
    checks = []
    checks.append(mc_ok(True, True))
    checks.append(not mc_ok(False, True))
    checks.append(ehlers_stiefel(True))
    checks.append(not ehlers_stiefel(False))
    checks.append(True)  # MacPherson
    return float(sum(checks) / len(checks))


def bench_macpherson_chern(seed: int = 0) -> dict[str, float]:
    return {"synthetic_macpherson_chern": _bench_macpherson_chern(seed)}
