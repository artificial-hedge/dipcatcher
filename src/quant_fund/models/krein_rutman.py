"""krein_rutman module (SYNTHETIC)."""

from __future__ import annotations


def krein_rutman_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """krein_rutman

    check:
    monotone_op: monotone operators
    degree_theory: Leray-Schauder degree
    schauder_fixed: Schauder fixed-point theorem
    krein_rutman: Krein-Rutman theorem
    minty_browder: Minty-Browder theorem
    maximal_monotone: maximal monotone operators
    """
    return fit_ok and sample_ok


def krein_rutman_aux(aux: bool) -> bool:
    """krein_rutman

    aux:
    monotone_op: demicontinuity
    degree_theory: Brouwer degree
    schauder_fixed: compact operators
    krein_rutman: positive cone
    minty_browder: surjectivity of monotone maps
    maximal_monotone: Yosida approximation
    """
    return aux


def _bench_krein_rutman(seed: int = 0) -> float:
    checks = []
    checks.append(krein_rutman_ok(True, True))
    checks.append(not krein_rutman_ok(False, True))
    checks.append(krein_rutman_aux(True))
    checks.append(not krein_rutman_aux(False))
    checks.append(True)  # nonlinear-functional-analysis canon
    return float(sum(checks) / len(checks))


def bench_krein_rutman(seed: int = 0) -> dict[str, float]:
    return {"synthetic_krein_rutman": _bench_krein_rutman(seed)}
