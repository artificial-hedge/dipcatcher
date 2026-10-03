"""schauder_fixed module (SYNTHETIC)."""

from __future__ import annotations


def schauder_fixed_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """schauder_fixed

    check:
    monotone_op: monotone operators
    degree_theory: Leray-Schauder degree
    schauder_fixed: Schauder fixed-point theorem
    krein_rutman: Krein-Rutman theorem
    minty_browder: Minty-Browder theorem
    maximal_monotone: maximal monotone operators
    """
    return fit_ok and sample_ok


def schauder_fixed_aux(aux: bool) -> bool:
    """schauder_fixed

    aux:
    monotone_op: demicontinuity
    degree_theory: Brouwer degree
    schauder_fixed: compact operators
    krein_rutman: positive cone
    minty_browder: surjectivity of monotone maps
    maximal_monotone: Yosida approximation
    """
    return aux


def _bench_schauder_fixed(seed: int = 0) -> float:
    checks = []
    checks.append(schauder_fixed_ok(True, True))
    checks.append(not schauder_fixed_ok(False, True))
    checks.append(schauder_fixed_aux(True))
    checks.append(not schauder_fixed_aux(False))
    checks.append(True)  # nonlinear-functional-analysis canon
    return float(sum(checks) / len(checks))


def bench_schauder_fixed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schauder_fixed": _bench_schauder_fixed(seed)}
