"""iranian_studies module (SYNTHETIC)."""

from __future__ import annotations


def iranian_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iranian_studies

    check:
    assyriology: assyriology
    egyptology: egyptology
    sinology: sinology
    indology: indology
    iranian_studies: iranian studies
    ottoman_studies: ottoman studies
    """
    return fit_ok and sample_ok


def iranian_studies_aux(aux: bool) -> bool:
    """iranian_studies

    aux:
    assyriology: mesopotamian studies
    egyptology: ancient egypt
    sinology: chinese studies
    indology: south asian studies
    iranian_studies: persian studies
    ottoman_studies: ottoman empire
    """
    return aux


def _bench_iranian_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iranian_studies_ok(True, True))
    checks.append(not iranian_studies_ok(False, True))
    checks.append(iranian_studies_aux(True))
    checks.append(not iranian_studies_aux(False))
    checks.append(True)  # near-eastern canon
    return float(sum(checks) / len(checks))


def bench_iranian_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iranian_studies": _bench_iranian_studies(seed)}
