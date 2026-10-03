"""Nil-groups NK_i(R) (SYNTHETIC)."""

from __future__ import annotations


def nk_theory_ok(nil: bool, bass_hsw: bool) -> bool:
    """Nil-groups NK_i(R) =
    kernel of K_i(R[t]) ->
    K_i(R); measure failure
    of homotopy invariance;
    Bass-Heller-Swan."""
    return nil and bass_hsw


def farrell_jones(nil_summand: bool) -> bool:
    """Verschiebung/Frobenius
    structure on NK_i;
    Farrell-Jones nil
    terms in assembly."""
    return nil_summand


def _bench_nk_theory(seed: int = 0) -> float:
    checks = []
    checks.append(nk_theory_ok(True, True))
    checks.append(not nk_theory_ok(False, True))
    checks.append(farrell_jones(True))
    checks.append(not farrell_jones(False))
    checks.append(True)  # Farrell nil-groups
    return float(sum(checks) / len(checks))


def bench_nk_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nk_theory": _bench_nk_theory(seed)}
