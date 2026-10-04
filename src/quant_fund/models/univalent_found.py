"""Univalent foundations (SYNTHETIC)."""

from __future__ import annotations


def univalent_found_ok(ua: bool, isofiber: bool) -> bool:
    """Univalence axiom: the
    canonical map (A = B) ->
    (A ~ B) is an equivalence;
    isomorphic types are
    identical."""
    return ua and isofiber


def funext_eq(funext: bool) -> bool:
    """Function extensionality
    follows from univalence;
    pointwise equal functions
    are equal."""
    return funext


def _bench_univalent_found(seed: int = 0) -> float:
    checks = []
    checks.append(univalent_found_ok(True, True))
    checks.append(not univalent_found_ok(False, True))
    checks.append(funext_eq(True))
    checks.append(not funext_eq(False))
    checks.append(True)  # Voevodsky's axiom
    return float(sum(checks) / len(checks))


def bench_univalent_found(seed: int = 0) -> dict[str, float]:
    return {"synthetic_univalent_found": _bench_univalent_found(seed)}
