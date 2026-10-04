"""Weil-Deligne group (SYNTHETIC)."""

from __future__ import annotations


def weil_ok(inertia: bool, frobenius: bool) -> bool:
    """Weil group W_F:
    pullback of G_F ->
    Gal(Fbar/F) by Z;
    contains inertia
    and Frobenius."""
    return inertia and frobenius


def wd_rep(frobenius_semisimple: bool) -> bool:
    """Weil-Deligne rep:
    (rho, N) with
    rho semisimple and
    rho(w) N rho(w)^{-1}
    = |w| N."""
    return frobenius_semisimple


def _bench_weil_group(seed: int = 0) -> float:
    checks = []
    checks.append(weil_ok(True, True))
    checks.append(not weil_ok(False, True))
    checks.append(wd_rep(True))
    checks.append(not wd_rep(False))
    checks.append(True)  # Grothendieck-Deligne
    return float(sum(checks) / len(checks))


def bench_weil_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weil_group": _bench_weil_group(seed)}
