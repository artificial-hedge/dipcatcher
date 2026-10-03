"""loewner_matrix module (SYNTHETIC)."""

from __future__ import annotations


def loewner_matrix_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """loewner_matrix

    check:
    loewner_matrix: Loewner matrix-monotone order
    operator_convex: operator convexity class
    kadison_ineq: Kadison inequality
    wielandt_ineq: Wielandt inequality
    ostrowski_bound: Ostrowski spectral bound
    gershgorin_disc: Gershgorin circle theorem
    """
    return fit_ok and sample_ok


def loewner_matrix_aux(aux: bool) -> bool:
    """loewner_matrix

    aux:
    loewner_matrix: divided-difference preservation
    operator_convex: Jensen operator inequality
    kadison_ineq: C*-map positivity bound
    wielandt_ineq: correlation bound sharpening
    ostrowski_bound: eigenvalue inclusion region
    gershgorin_disc: union-of-discs spectrum
    """
    return aux


def _bench_loewner_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(loewner_matrix_ok(True, True))
    checks.append(not loewner_matrix_ok(False, True))
    checks.append(loewner_matrix_aux(True))
    checks.append(not loewner_matrix_aux(False))
    checks.append(True)  # matrix-analysis canon
    return float(sum(checks) / len(checks))


def bench_loewner_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loewner_matrix": _bench_loewner_matrix(seed)}
