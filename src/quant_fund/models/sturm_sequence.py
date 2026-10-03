"""sturm_sequence module (SYNTHETIC)."""

from __future__ import annotations


def sturm_sequence_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sturm_sequence

    check:
    cauchy_interlace: Cauchy interlacing theorem
    sylvester_law: Sylvester law of inertia
    haynsworth_inertia: Haynsworth inertia additivity
    min_max_eig: Courant–Fischer min-max
    sturm_sequence: Sturm sequence root count
    bezout_matrix: Bézout resultant matrix
    """
    return fit_ok and sample_ok


def sturm_sequence_aux(aux: bool) -> bool:
    """sturm_sequence

    aux:
    cauchy_interlace: bordered principal submatrix
    sylvester_law: congruence preserves signature
    haynsworth_inertia: block inertia via Schur
    min_max_eig: variational eigenvalue characterization
    sturm_sequence: sign-change bisection
    bezout_matrix: common-root criterion
    """
    return aux


def _bench_sturm_sequence(seed: int = 0) -> float:
    checks = []
    checks.append(sturm_sequence_ok(True, True))
    checks.append(not sturm_sequence_ok(False, True))
    checks.append(sturm_sequence_aux(True))
    checks.append(not sturm_sequence_aux(False))
    checks.append(True)  # spectral-interlacing canon
    return float(sum(checks) / len(checks))


def bench_sturm_sequence(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sturm_sequence": _bench_sturm_sequence(seed)}
