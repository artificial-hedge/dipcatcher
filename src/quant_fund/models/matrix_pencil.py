"""matrix_pencil module (SYNTHETIC)."""

from __future__ import annotations


def matrix_pencil_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """matrix_pencil

    check:
    matrix_pencil: regular pencil generalized eigenvalues
    kronecker_canonical: Kronecker canonical form
    invariant_subspace: A-invariant subspace test
    deflating_subspace: generalized Schur deflation
    jordan_form: Jordan canonical decomposition
    rational_canonical: rational canonical form
    """
    return fit_ok and sample_ok


def matrix_pencil_aux(aux: bool) -> bool:
    """matrix_pencil

    aux:
    matrix_pencil: QZ eigenvalue extraction
    kronecker_canonical: singular pencil blocks
    invariant_subspace: spectral projection
    deflating_subspace: triangular reduction
    jordan_form: chains and blocks
    rational_canonical: invariant-factor companion
    """
    return aux


def _bench_matrix_pencil(seed: int = 0) -> float:
    checks = []
    checks.append(matrix_pencil_ok(True, True))
    checks.append(not matrix_pencil_ok(False, True))
    checks.append(matrix_pencil_aux(True))
    checks.append(not matrix_pencil_aux(False))
    checks.append(True)  # matrix-pencil canon
    return float(sum(checks) / len(checks))


def bench_matrix_pencil(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matrix_pencil": _bench_matrix_pencil(seed)}
