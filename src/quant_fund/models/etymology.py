"""etymology module (SYNTHETIC)."""

from __future__ import annotations


def etymology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """etymology

    check:
    contact_linguistics: contact linguistics
    descriptive_linguistics: descriptive linguistics
    philological_studies: philology
    etymology: etymology
    dialectometry: dialectometry
    lexicography: lexicography
    """
    return fit_ok and sample_ok


def etymology_aux(aux: bool) -> bool:
    """etymology

    aux:
    contact_linguistics: language contact
    descriptive_linguistics: language description
    philological_studies: textual study
    etymology: word origins
    dialectometry: dialect distances
    lexicography: dictionary making
    """
    return aux


def _bench_etymology(seed: int = 0) -> float:
    checks = []
    checks.append(etymology_ok(True, True))
    checks.append(not etymology_ok(False, True))
    checks.append(etymology_aux(True))
    checks.append(not etymology_aux(False))
    checks.append(True)  # linguistics-5 canon
    return float(sum(checks) / len(checks))


def bench_etymology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etymology": _bench_etymology(seed)}
