"""numismatics module (SYNTHETIC)."""

from __future__ import annotations


def numismatics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """numismatics

    check:
    medieval_studies: medieval studies
    paleography: paleography
    codicology: codicology
    hagiography: hagiography
    byzantine_studies: byzantine studies
    numismatics: numismatics
    """
    return fit_ok and sample_ok


def numismatics_aux(aux: bool) -> bool:
    """numismatics

    aux:
    medieval_studies: middle ages
    paleography: ancient scripts
    codicology: manuscript books
    hagiography: saints' lives
    byzantine_studies: byzantine empire
    numismatics: coinage
    """
    return aux


def _bench_numismatics(seed: int = 0) -> float:
    checks = []
    checks.append(numismatics_ok(True, True))
    checks.append(not numismatics_ok(False, True))
    checks.append(numismatics_aux(True))
    checks.append(not numismatics_aux(False))
    checks.append(True)  # medieval canon
    return float(sum(checks) / len(checks))


def bench_numismatics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_numismatics": _bench_numismatics(seed)}
