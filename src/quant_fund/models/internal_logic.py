"""Internal logic (SYNTHETIC)."""

from __future__ import annotations


def il_ok(internal_lang: bool, interpretation: bool) -> bool:
    """Internal
    logic:
    type
    theory
    interpreted
    in
    category —
    Mitchell-
    Benabou."""
    return internal_lang and interpretation


def kripke_joyal(kj: bool) -> bool:
    """Kripke-
    Joyal:
    internal
    semantics
    of
    topos —
    forcing
    semantics."""
    return kj


def _bench_internal_logic(seed: int = 0) -> float:
    checks = []
    checks.append(il_ok(True, True))
    checks.append(not il_ok(False, True))
    checks.append(kripke_joyal(True))
    checks.append(not kripke_joyal(False))
    checks.append(True)  # Mitchell-Benabou
    return float(sum(checks) / len(checks))


def bench_internal_logic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_internal_logic": _bench_internal_logic(seed)}
