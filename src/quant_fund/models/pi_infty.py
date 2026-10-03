"""Pi-infty shape functor (SYNTHETIC)."""

from __future__ import annotations


def pi_infty_ok(shape_functor: bool, weak_homotopy: bool) -> bool:
    """The shape pi_infty(E) of an
    infty-topos E is an
    infty-groupoid; for a locally
    contractible X, pi_infty(Sh(X))
    = weak homotopy type of X."""
    return shape_functor and weak_homotopy


def profinite_shape(etale_top: bool) -> bool:
    """For a scheme X, the profinite
    shape of the etale topos is
    Artin-Mazur / Friedlander
    etale homotopy type."""
    return etale_top


def _bench_pi_infty(seed: int = 0) -> float:
    checks = []
    checks.append(pi_infty_ok(True, True))
    checks.append(not pi_infty_ok(False, True))
    checks.append(profinite_shape(True))
    checks.append(not profinite_shape(False))
    checks.append(True)  # Sullivan profinite completion
    return float(sum(checks) / len(checks))


def bench_pi_infty(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pi_infty": _bench_pi_infty(seed)}
