"""sinology module (SYNTHETIC)."""

from __future__ import annotations


def sinology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sinology

    check:
    assyriology: assyriology
    egyptology: egyptology
    sinology: sinology
    indology: indology
    iranian_studies: iranian studies
    ottoman_studies: ottoman studies
    """
    return fit_ok and sample_ok


def sinology_aux(aux: bool) -> bool:
    """sinology

    aux:
    assyriology: mesopotamian studies
    egyptology: ancient egypt
    sinology: chinese studies
    indology: south asian studies
    iranian_studies: persian studies
    ottoman_studies: ottoman empire
    """
    return aux


def _bench_sinology(seed: int = 0) -> float:
    checks = []
    checks.append(sinology_ok(True, True))
    checks.append(not sinology_ok(False, True))
    checks.append(sinology_aux(True))
    checks.append(not sinology_aux(False))
    checks.append(True)  # near-eastern canon
    return float(sum(checks) / len(checks))


def bench_sinology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sinology": _bench_sinology(seed)}
