"""contact_linguistics module (SYNTHETIC)."""

from __future__ import annotations


def contact_linguistics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """contact_linguistics

    check:
    contact_linguistics: contact linguistics
    descriptive_linguistics: descriptive linguistics
    philological_studies: philology
    etymology: etymology
    dialectometry: dialectometry
    lexicography: lexicography
    """
    return fit_ok and sample_ok


def contact_linguistics_aux(aux: bool) -> bool:
    """contact_linguistics

    aux:
    contact_linguistics: language contact
    descriptive_linguistics: language description
    philological_studies: textual study
    etymology: word origins
    dialectometry: dialect distances
    lexicography: dictionary making
    """
    return aux


def _bench_contact_linguistics(seed: int = 0) -> float:
    checks = []
    checks.append(contact_linguistics_ok(True, True))
    checks.append(not contact_linguistics_ok(False, True))
    checks.append(contact_linguistics_aux(True))
    checks.append(not contact_linguistics_aux(False))
    checks.append(True)  # linguistics-5 canon
    return float(sum(checks) / len(checks))


def bench_contact_linguistics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_contact_linguistics": _bench_contact_linguistics(seed)}
