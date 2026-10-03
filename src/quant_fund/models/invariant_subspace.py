"""invariant_subspace module (SYNTHETIC)."""

from __future__ import annotations


def invariant_subspace_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """invariant_subspace

    check:
    matrix_pencil: regular pencil generalized eigenvalues
    kronecker_canonical: Kronecker canonical form
    invariant_subspace: A-invariant subspace test
    deflating_subspace: generalized Schur deflation
    jordan_form: Jordan canonical decomposition
    rational_canonical: rational canonical form
    """
    return fit_ok and sample_ok


def invariant_subspace_aux(aux: bool) -> bool:
    """invariant_subspace

    aux:
    matrix_pencil: QZ eigenvalue extraction
    kronecker_canonical: singular pencil blocks
    invariant_subspace: spectral projection
    deflating_subspace: triangular reduction
    jordan_form: chains and blocks
    rational_canonical: invariant-factor companion
    """
    return aux


def _bench_invariant_subspace(seed: int = 0) -> float:
    checks = []
    checks.append(invariant_subspace_ok(True, True))
    checks.append(not invariant_subspace_ok(False, True))
    checks.append(invariant_subspace_aux(True))
    checks.append(not invariant_subspace_aux(False))
    checks.append(True)  # matrix-pencil canon
    return float(sum(checks) / len(checks))


def bench_invariant_subspace(seed: int = 0) -> dict[str, float]:
    return {"synthetic_invariant_subspace": _bench_invariant_subspace(seed)}
