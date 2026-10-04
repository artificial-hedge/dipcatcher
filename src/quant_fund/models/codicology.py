"""codicology module (SYNTHETIC)."""

from __future__ import annotations


def codicology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """codicology

    check:
    medieval_studies: medieval studies
    paleography: paleography
    codicology: codicology
    hagiography: hagiography
    byzantine_studies: byzantine studies
    numismatics: numismatics
    """
    return fit_ok and sample_ok


def codicology_aux(aux: bool) -> bool:
    """codicology

    aux:
    medieval_studies: middle ages
    paleography: ancient scripts
    codicology: manuscript books
    hagiography: saints' lives
    byzantine_studies: byzantine empire
    numismatics: coinage
    """
    return aux


def _bench_codicology(seed: int = 0) -> float:
    checks = []
    checks.append(codicology_ok(True, True))
    checks.append(not codicology_ok(False, True))
    checks.append(codicology_aux(True))
    checks.append(not codicology_aux(False))
    checks.append(True)  # medieval canon
    return float(sum(checks) / len(checks))


def bench_codicology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_codicology": _bench_codicology(seed)}
