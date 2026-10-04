"""dialectometry module (SYNTHETIC)."""

from __future__ import annotations


def dialectometry_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dialectometry

    check:
    contact_linguistics: contact linguistics
    descriptive_linguistics: descriptive linguistics
    philological_studies: philology
    etymology: etymology
    dialectometry: dialectometry
    lexicography: lexicography
    """
    return fit_ok and sample_ok


def dialectometry_aux(aux: bool) -> bool:
    """dialectometry

    aux:
    contact_linguistics: language contact
    descriptive_linguistics: language description
    philological_studies: textual study
    etymology: word origins
    dialectometry: dialect distances
    lexicography: dictionary making
    """
    return aux


def _bench_dialectometry(seed: int = 0) -> float:
    checks = []
    checks.append(dialectometry_ok(True, True))
    checks.append(not dialectometry_ok(False, True))
    checks.append(dialectometry_aux(True))
    checks.append(not dialectometry_aux(False))
    checks.append(True)  # linguistics-5 canon
    return float(sum(checks) / len(checks))


def bench_dialectometry(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dialectometry": _bench_dialectometry(seed)}
