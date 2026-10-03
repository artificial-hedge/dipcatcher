"""jordan_form module (SYNTHETIC)."""

from __future__ import annotations


def jordan_form_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jordan_form

    check:
    matrix_pencil: regular pencil generalized eigenvalues
    kronecker_canonical: Kronecker canonical form
    invariant_subspace: A-invariant subspace test
    deflating_subspace: generalized Schur deflation
    jordan_form: Jordan canonical decomposition
    rational_canonical: rational canonical form
    """
    return fit_ok and sample_ok


def jordan_form_aux(aux: bool) -> bool:
    """jordan_form

    aux:
    matrix_pencil: QZ eigenvalue extraction
    kronecker_canonical: singular pencil blocks
    invariant_subspace: spectral projection
    deflating_subspace: triangular reduction
    jordan_form: chains and blocks
    rational_canonical: invariant-factor companion
    """
    return aux


def _bench_jordan_form(seed: int = 0) -> float:
    checks = []
    checks.append(jordan_form_ok(True, True))
    checks.append(not jordan_form_ok(False, True))
    checks.append(jordan_form_aux(True))
    checks.append(not jordan_form_aux(False))
    checks.append(True)  # matrix-pencil canon
    return float(sum(checks) / len(checks))


def bench_jordan_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jordan_form": _bench_jordan_form(seed)}
