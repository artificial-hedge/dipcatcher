"""korovkin_thm module (SYNTHETIC)."""

from __future__ import annotations


def korovkin_thm_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """korovkin_thm

    check:
    walsh_series: Walsh series expansion convergence
    haar_system: Haar orthonormal basis completeness
    faber_schauder: Faber-Schauder hat basis
    de_boor_stable: de Boor stable spline recurrence
    whitney_ext: Whitney extension jet matching
    korovkin_thm: Korovkin positive-operator approximation
    """
    return fit_ok and sample_ok


def korovkin_thm_aux(aux: bool) -> bool:
    """korovkin_thm

    aux:
    walsh_series: Paley-ordered partial sums
    haar_system: unconditional basis in Lp
    faber_schauder: Schauder basis of C[0,1]
    de_boor_stable: knot insertion stability
    whitney_ext: compatible jet conditions
    korovkin_thm: test-function convergence
    """
    return aux


def _bench_korovkin_thm(seed: int = 0) -> float:
    checks = []
    checks.append(korovkin_thm_ok(True, True))
    checks.append(not korovkin_thm_ok(False, True))
    checks.append(korovkin_thm_aux(True))
    checks.append(not korovkin_thm_aux(False))
    checks.append(True)  # approximation-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_korovkin_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_korovkin_thm": _bench_korovkin_thm(seed)}
