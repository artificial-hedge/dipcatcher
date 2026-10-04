"""lexicography module (SYNTHETIC)."""

from __future__ import annotations


def lexicography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lexicography

    check:
    contact_linguistics: contact linguistics
    descriptive_linguistics: descriptive linguistics
    philological_studies: philology
    etymology: etymology
    dialectometry: dialectometry
    lexicography: lexicography
    """
    return fit_ok and sample_ok


def lexicography_aux(aux: bool) -> bool:
    """lexicography

    aux:
    contact_linguistics: language contact
    descriptive_linguistics: language description
    philological_studies: textual study
    etymology: word origins
    dialectometry: dialect distances
    lexicography: dictionary making
    """
    return aux


def _bench_lexicography(seed: int = 0) -> float:
    checks = []
    checks.append(lexicography_ok(True, True))
    checks.append(not lexicography_ok(False, True))
    checks.append(lexicography_aux(True))
    checks.append(not lexicography_aux(False))
    checks.append(True)  # linguistics-5 canon
    return float(sum(checks) / len(checks))


def bench_lexicography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lexicography": _bench_lexicography(seed)}
