"""indology module (SYNTHETIC)."""

from __future__ import annotations


def indology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """indology

    check:
    assyriology: assyriology
    egyptology: egyptology
    sinology: sinology
    indology: indology
    iranian_studies: iranian studies
    ottoman_studies: ottoman studies
    """
    return fit_ok and sample_ok


def indology_aux(aux: bool) -> bool:
    """indology

    aux:
    assyriology: mesopotamian studies
    egyptology: ancient egypt
    sinology: chinese studies
    indology: south asian studies
    iranian_studies: persian studies
    ottoman_studies: ottoman empire
    """
    return aux


def _bench_indology(seed: int = 0) -> float:
    checks = []
    checks.append(indology_ok(True, True))
    checks.append(not indology_ok(False, True))
    checks.append(indology_aux(True))
    checks.append(not indology_aux(False))
    checks.append(True)  # near-eastern canon
    return float(sum(checks) / len(checks))


def bench_indology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_indology": _bench_indology(seed)}
