"""egyptology module (SYNTHETIC)."""

from __future__ import annotations


def egyptology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """egyptology

    check:
    assyriology: assyriology
    egyptology: egyptology
    sinology: sinology
    indology: indology
    iranian_studies: iranian studies
    ottoman_studies: ottoman studies
    """
    return fit_ok and sample_ok


def egyptology_aux(aux: bool) -> bool:
    """egyptology

    aux:
    assyriology: mesopotamian studies
    egyptology: ancient egypt
    sinology: chinese studies
    indology: south asian studies
    iranian_studies: persian studies
    ottoman_studies: ottoman empire
    """
    return aux


def _bench_egyptology(seed: int = 0) -> float:
    checks = []
    checks.append(egyptology_ok(True, True))
    checks.append(not egyptology_ok(False, True))
    checks.append(egyptology_aux(True))
    checks.append(not egyptology_aux(False))
    checks.append(True)  # near-eastern canon
    return float(sum(checks) / len(checks))


def bench_egyptology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_egyptology": _bench_egyptology(seed)}
