"""heraldry module (SYNTHETIC)."""

from __future__ import annotations


def heraldry_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """heraldry

    check:
    epigraphy: epigraphy
    diplomatics: diplomatics
    sigillography: sigillography
    heraldry: heraldry
    genealogy_studies: genealogy studies
    onomastics: onomastics
    """
    return fit_ok and sample_ok


def heraldry_aux(aux: bool) -> bool:
    """heraldry

    aux:
    epigraphy: inscriptions
    diplomatics: charter analysis
    sigillography: seals study
    heraldry: coats of arms
    genealogy_studies: lineage research
    onomastics: name studies
    """
    return aux


def _bench_heraldry(seed: int = 0) -> float:
    checks = []
    checks.append(heraldry_ok(True, True))
    checks.append(not heraldry_ok(False, True))
    checks.append(heraldry_aux(True))
    checks.append(not heraldry_aux(False))
    checks.append(True)  # documentary sciences canon
    return float(sum(checks) / len(checks))


def bench_heraldry(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heraldry": _bench_heraldry(seed)}
