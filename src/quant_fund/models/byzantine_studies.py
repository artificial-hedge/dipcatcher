"""byzantine_studies module (SYNTHETIC)."""

from __future__ import annotations


def byzantine_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """byzantine_studies

    check:
    medieval_studies: medieval studies
    paleography: paleography
    codicology: codicology
    hagiography: hagiography
    byzantine_studies: byzantine studies
    numismatics: numismatics
    """
    return fit_ok and sample_ok


def byzantine_studies_aux(aux: bool) -> bool:
    """byzantine_studies

    aux:
    medieval_studies: middle ages
    paleography: ancient scripts
    codicology: manuscript books
    hagiography: saints' lives
    byzantine_studies: byzantine empire
    numismatics: coinage
    """
    return aux


def _bench_byzantine_studies(seed: int = 0) -> float:
    checks = []
    checks.append(byzantine_studies_ok(True, True))
    checks.append(not byzantine_studies_ok(False, True))
    checks.append(byzantine_studies_aux(True))
    checks.append(not byzantine_studies_aux(False))
    checks.append(True)  # medieval canon
    return float(sum(checks) / len(checks))


def bench_byzantine_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_byzantine_studies": _bench_byzantine_studies(seed)}
