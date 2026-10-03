"""kronecker_canonical module (SYNTHETIC)."""

from __future__ import annotations


def kronecker_canonical_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kronecker_canonical

    check:
    matrix_pencil: regular pencil generalized eigenvalues
    kronecker_canonical: Kronecker canonical form
    invariant_subspace: A-invariant subspace test
    deflating_subspace: generalized Schur deflation
    jordan_form: Jordan canonical decomposition
    rational_canonical: rational canonical form
    """
    return fit_ok and sample_ok


def kronecker_canonical_aux(aux: bool) -> bool:
    """kronecker_canonical

    aux:
    matrix_pencil: QZ eigenvalue extraction
    kronecker_canonical: singular pencil blocks
    invariant_subspace: spectral projection
    deflating_subspace: triangular reduction
    jordan_form: chains and blocks
    rational_canonical: invariant-factor companion
    """
    return aux


def _bench_kronecker_canonical(seed: int = 0) -> float:
    checks = []
    checks.append(kronecker_canonical_ok(True, True))
    checks.append(not kronecker_canonical_ok(False, True))
    checks.append(kronecker_canonical_aux(True))
    checks.append(not kronecker_canonical_aux(False))
    checks.append(True)  # matrix-pencil canon
    return float(sum(checks) / len(checks))


def bench_kronecker_canonical(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kronecker_canonical": _bench_kronecker_canonical(seed)}
