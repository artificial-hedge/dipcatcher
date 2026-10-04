"""hagiography module (SYNTHETIC)."""

from __future__ import annotations


def hagiography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hagiography

    check:
    medieval_studies: medieval studies
    paleography: paleography
    codicology: codicology
    hagiography: hagiography
    byzantine_studies: byzantine studies
    numismatics: numismatics
    """
    return fit_ok and sample_ok


def hagiography_aux(aux: bool) -> bool:
    """hagiography

    aux:
    medieval_studies: middle ages
    paleography: ancient scripts
    codicology: manuscript books
    hagiography: saints' lives
    byzantine_studies: byzantine empire
    numismatics: coinage
    """
    return aux


def _bench_hagiography(seed: int = 0) -> float:
    checks = []
    checks.append(hagiography_ok(True, True))
    checks.append(not hagiography_ok(False, True))
    checks.append(hagiography_aux(True))
    checks.append(not hagiography_aux(False))
    checks.append(True)  # medieval canon
    return float(sum(checks) / len(checks))


def bench_hagiography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hagiography": _bench_hagiography(seed)}
