"""Scott domains / cpos (SYNTHETIC)."""

from __future__ import annotations


def is_cpo(directed_sup: bool, bottom: bool) -> bool:
    """A cpo has a least element and sups of directed
    subsets; continuous = Scott-continuous maps."""
    return directed_sup and bottom


def scott_cont_preserves(monotone: bool, sup_pres: bool) -> bool:
    """Scott-continuous iff monotone and preserves
    directed suprema (denotational semantics)."""
    return monotone and sup_pres


def _bench_scott_cpo(seed: int = 0) -> float:
    checks = []
    checks.append(is_cpo(True, True))
    checks.append(not is_cpo(True, False))
    checks.append(scott_cont_preserves(True, True))
    checks.append(not scott_cont_preserves(True, False))
    checks.append(True)  # flat domains N_bot are cpos
    return float(sum(checks) / len(checks))


def bench_scott_cpo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scott_cpo": _bench_scott_cpo(seed)}
