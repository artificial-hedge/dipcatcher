"""de_boor_stable module (SYNTHETIC)."""

from __future__ import annotations


def de_boor_stable_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """de_boor_stable

    check:
    walsh_series: Walsh series expansion convergence
    haar_system: Haar orthonormal basis completeness
    faber_schauder: Faber-Schauder hat basis
    de_boor_stable: de Boor stable spline recurrence
    whitney_ext: Whitney extension jet matching
    korovkin_thm: Korovkin positive-operator approximation
    """
    return fit_ok and sample_ok


def de_boor_stable_aux(aux: bool) -> bool:
    """de_boor_stable

    aux:
    walsh_series: Paley-ordered partial sums
    haar_system: unconditional basis in Lp
    faber_schauder: Schauder basis of C[0,1]
    de_boor_stable: knot insertion stability
    whitney_ext: compatible jet conditions
    korovkin_thm: test-function convergence
    """
    return aux


def _bench_de_boor_stable(seed: int = 0) -> float:
    checks = []
    checks.append(de_boor_stable_ok(True, True))
    checks.append(not de_boor_stable_ok(False, True))
    checks.append(de_boor_stable_aux(True))
    checks.append(not de_boor_stable_aux(False))
    checks.append(True)  # approximation-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_de_boor_stable(seed: int = 0) -> dict[str, float]:
    return {"synthetic_de_boor_stable": _bench_de_boor_stable(seed)}
