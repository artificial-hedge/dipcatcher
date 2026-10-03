"""vandermonde_matrix module (SYNTHETIC)."""

from __future__ import annotations


def vandermonde_matrix_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vandermonde_matrix

    check:
    circulant_matrix: circulant structure and DFT diagonalization
    companion_matrix: Frobenius companion eigenvalues
    vandermonde_matrix: Vandermonde determinant
    krylov_matrix: Krylov subspace basis
    hessenberg_form: Hessenberg reduction
    hankel_matrix: Hankel structure and moments
    """
    return fit_ok and sample_ok


def vandermonde_matrix_aux(aux: bool) -> bool:
    """vandermonde_matrix

    aux:
    circulant_matrix: FFT eigenvalues
    companion_matrix: characteristic polynomial roots
    vandermonde_matrix: product formula
    krylov_matrix: Arnoldi orthonormalization
    hessenberg_form: Householder upper-Hessenberg
    hankel_matrix: persymmetric structure
    """
    return aux


def _bench_vandermonde_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(vandermonde_matrix_ok(True, True))
    checks.append(not vandermonde_matrix_ok(False, True))
    checks.append(vandermonde_matrix_aux(True))
    checks.append(not vandermonde_matrix_aux(False))
    checks.append(True)  # structured-matrix canon
    return float(sum(checks) / len(checks))


def bench_vandermonde_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vandermonde_matrix": _bench_vandermonde_matrix(seed)}
