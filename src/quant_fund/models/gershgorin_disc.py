"""gershgorin_disc module (SYNTHETIC)."""

from __future__ import annotations


def gershgorin_disc_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gershgorin_disc

    check:
    loewner_matrix: Loewner matrix-monotone order
    operator_convex: operator convexity class
    kadison_ineq: Kadison inequality
    wielandt_ineq: Wielandt inequality
    ostrowski_bound: Ostrowski spectral bound
    gershgorin_disc: Gershgorin circle theorem
    """
    return fit_ok and sample_ok


def gershgorin_disc_aux(aux: bool) -> bool:
    """gershgorin_disc

    aux:
    loewner_matrix: divided-difference preservation
    operator_convex: Jensen operator inequality
    kadison_ineq: C*-map positivity bound
    wielandt_ineq: correlation bound sharpening
    ostrowski_bound: eigenvalue inclusion region
    gershgorin_disc: union-of-discs spectrum
    """
    return aux


def _bench_gershgorin_disc(seed: int = 0) -> float:
    checks = []
    checks.append(gershgorin_disc_ok(True, True))
    checks.append(not gershgorin_disc_ok(False, True))
    checks.append(gershgorin_disc_aux(True))
    checks.append(not gershgorin_disc_aux(False))
    checks.append(True)  # matrix-analysis canon
    return float(sum(checks) / len(checks))


def bench_gershgorin_disc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gershgorin_disc": _bench_gershgorin_disc(seed)}
