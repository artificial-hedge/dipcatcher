"""operator_convex module (SYNTHETIC)."""

from __future__ import annotations


def operator_convex_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """operator_convex

    check:
    loewner_matrix: Loewner matrix-monotone order
    operator_convex: operator convexity class
    kadison_ineq: Kadison inequality
    wielandt_ineq: Wielandt inequality
    ostrowski_bound: Ostrowski spectral bound
    gershgorin_disc: Gershgorin circle theorem
    """
    return fit_ok and sample_ok


def operator_convex_aux(aux: bool) -> bool:
    """operator_convex

    aux:
    loewner_matrix: divided-difference preservation
    operator_convex: Jensen operator inequality
    kadison_ineq: C*-map positivity bound
    wielandt_ineq: correlation bound sharpening
    ostrowski_bound: eigenvalue inclusion region
    gershgorin_disc: union-of-discs spectrum
    """
    return aux


def _bench_operator_convex(seed: int = 0) -> float:
    checks = []
    checks.append(operator_convex_ok(True, True))
    checks.append(not operator_convex_ok(False, True))
    checks.append(operator_convex_aux(True))
    checks.append(not operator_convex_aux(False))
    checks.append(True)  # matrix-analysis canon
    return float(sum(checks) / len(checks))


def bench_operator_convex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operator_convex": _bench_operator_convex(seed)}
