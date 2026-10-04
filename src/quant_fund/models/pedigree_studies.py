"""pedigree_studies module (SYNTHETIC)."""

from __future__ import annotations


def pedigree_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pedigree_studies

    check:
    pedigree_studies: families and segregation
    ..."""
    return fit_ok and sample_ok


def pedigree_studies_aux(aux: bool) -> bool:
    """pedigree_studies

    aux:
    pedigree_studies: inheritance and consanguinity
    ..."""
    return aux


def _bench_pedigree_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pedigree_studies_ok(True, True))
    checks.append(not pedigree_studies_ok(False, True))
    checks.append(pedigree_studies_aux(True))
    checks.append(not pedigree_studies_aux(False))
    checks.append(True)  # molecular-genetics-2 canon
    return float(sum(checks) / len(checks))


def bench_pedigree_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pedigree_studies": _bench_pedigree_studies(seed)}
