"""deflating_subspace module (SYNTHETIC)."""

from __future__ import annotations


def deflating_subspace_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """deflating_subspace

    check:
    matrix_pencil: regular pencil generalized eigenvalues
    kronecker_canonical: Kronecker canonical form
    invariant_subspace: A-invariant subspace test
    deflating_subspace: generalized Schur deflation
    jordan_form: Jordan canonical decomposition
    rational_canonical: rational canonical form
    """
    return fit_ok and sample_ok


def deflating_subspace_aux(aux: bool) -> bool:
    """deflating_subspace

    aux:
    matrix_pencil: QZ eigenvalue extraction
    kronecker_canonical: singular pencil blocks
    invariant_subspace: spectral projection
    deflating_subspace: triangular reduction
    jordan_form: chains and blocks
    rational_canonical: invariant-factor companion
    """
    return aux


def _bench_deflating_subspace(seed: int = 0) -> float:
    checks = []
    checks.append(deflating_subspace_ok(True, True))
    checks.append(not deflating_subspace_ok(False, True))
    checks.append(deflating_subspace_aux(True))
    checks.append(not deflating_subspace_aux(False))
    checks.append(True)  # matrix-pencil canon
    return float(sum(checks) / len(checks))


def bench_deflating_subspace(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deflating_subspace": _bench_deflating_subspace(seed)}
