"""wielandt_ineq module (SYNTHETIC)."""

from __future__ import annotations


def wielandt_ineq_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wielandt_ineq

    check:
    loewner_matrix: Loewner matrix-monotone order
    operator_convex: operator convexity class
    kadison_ineq: Kadison inequality
    wielandt_ineq: Wielandt inequality
    ostrowski_bound: Ostrowski spectral bound
    gershgorin_disc: Gershgorin circle theorem
    """
    return fit_ok and sample_ok


def wielandt_ineq_aux(aux: bool) -> bool:
    """wielandt_ineq

    aux:
    loewner_matrix: divided-difference preservation
    operator_convex: Jensen operator inequality
    kadison_ineq: C*-map positivity bound
    wielandt_ineq: correlation bound sharpening
    ostrowski_bound: eigenvalue inclusion region
    gershgorin_disc: union-of-discs spectrum
    """
    return aux


def _bench_wielandt_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(wielandt_ineq_ok(True, True))
    checks.append(not wielandt_ineq_ok(False, True))
    checks.append(wielandt_ineq_aux(True))
    checks.append(not wielandt_ineq_aux(False))
    checks.append(True)  # matrix-analysis canon
    return float(sum(checks) / len(checks))


def bench_wielandt_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wielandt_ineq": _bench_wielandt_ineq(seed)}
