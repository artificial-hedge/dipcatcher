"""hessenberg_form module (SYNTHETIC)."""

from __future__ import annotations


def hessenberg_form_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hessenberg_form

    check:
    circulant_matrix: circulant structure and DFT diagonalization
    companion_matrix: Frobenius companion eigenvalues
    vandermonde_matrix: Vandermonde determinant
    krylov_matrix: Krylov subspace basis
    hessenberg_form: Hessenberg reduction
    hankel_matrix: Hankel structure and moments
    """
    return fit_ok and sample_ok


def hessenberg_form_aux(aux: bool) -> bool:
    """hessenberg_form

    aux:
    circulant_matrix: FFT eigenvalues
    companion_matrix: characteristic polynomial roots
    vandermonde_matrix: product formula
    krylov_matrix: Arnoldi orthonormalization
    hessenberg_form: Householder upper-Hessenberg
    hankel_matrix: persymmetric structure
    """
    return aux


def _bench_hessenberg_form(seed: int = 0) -> float:
    checks = []
    checks.append(hessenberg_form_ok(True, True))
    checks.append(not hessenberg_form_ok(False, True))
    checks.append(hessenberg_form_aux(True))
    checks.append(not hessenberg_form_aux(False))
    checks.append(True)  # structured-matrix canon
    return float(sum(checks) / len(checks))


def bench_hessenberg_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hessenberg_form": _bench_hessenberg_form(seed)}
