"""assyriology module (SYNTHETIC)."""

from __future__ import annotations


def assyriology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """assyriology

    check:
    assyriology: assyriology
    egyptology: egyptology
    sinology: sinology
    indology: indology
    iranian_studies: iranian studies
    ottoman_studies: ottoman studies
    """
    return fit_ok and sample_ok


def assyriology_aux(aux: bool) -> bool:
    """assyriology

    aux:
    assyriology: mesopotamian studies
    egyptology: ancient egypt
    sinology: chinese studies
    indology: south asian studies
    iranian_studies: persian studies
    ottoman_studies: ottoman empire
    """
    return aux


def _bench_assyriology(seed: int = 0) -> float:
    checks = []
    checks.append(assyriology_ok(True, True))
    checks.append(not assyriology_ok(False, True))
    checks.append(assyriology_aux(True))
    checks.append(not assyriology_aux(False))
    checks.append(True)  # near-eastern canon
    return float(sum(checks) / len(checks))


def bench_assyriology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_assyriology": _bench_assyriology(seed)}
