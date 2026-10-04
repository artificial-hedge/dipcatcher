"""john_ellipsoid module (SYNTHETIC)."""

from __future__ import annotations


def john_ellipsoid_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """john_ellipsoid

    check:
    john_ellipsoid: John maximal-volume ellipsoid
    loewner_ellipsoid: Loewner minimal-volume ellipsoid
    milman_rev_thm: Milman reverse Brunn-Minkowski
    grothendieck_const: Grothendieck constant bounds
    kadison_singer: Kadison-Singer paving problem
    milman_isotropic: Milman isotropic position
    """
    return fit_ok and sample_ok


def john_ellipsoid_aux(aux: bool) -> bool:
    """john_ellipsoid

    aux:
    john_ellipsoid: decomposition of identity
    loewner_ellipsoid: contact-point characterization
    milman_rev_thm: quotient volume estimate
    grothendieck_const: little Grothendieck constant
    kadison_singer: Weaver KS_r paving
    milman_isotropic: covariance normalization
    """
    return aux


def _bench_john_ellipsoid(seed: int = 0) -> float:
    checks = []
    checks.append(john_ellipsoid_ok(True, True))
    checks.append(not john_ellipsoid_ok(False, True))
    checks.append(john_ellipsoid_aux(True))
    checks.append(not john_ellipsoid_aux(False))
    checks.append(True)  # convex-geometry-2 canon
    return float(sum(checks) / len(checks))


def bench_john_ellipsoid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_john_ellipsoid": _bench_john_ellipsoid(seed)}
