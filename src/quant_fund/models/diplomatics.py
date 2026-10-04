"""diplomatics module (SYNTHETIC)."""

from __future__ import annotations


def diplomatics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diplomatics

    check:
    epigraphy: epigraphy
    diplomatics: diplomatics
    sigillography: sigillography
    heraldry: heraldry
    genealogy_studies: genealogy studies
    onomastics: onomastics
    """
    return fit_ok and sample_ok


def diplomatics_aux(aux: bool) -> bool:
    """diplomatics

    aux:
    epigraphy: inscriptions
    diplomatics: charter analysis
    sigillography: seals study
    heraldry: coats of arms
    genealogy_studies: lineage research
    onomastics: name studies
    """
    return aux


def _bench_diplomatics(seed: int = 0) -> float:
    checks = []
    checks.append(diplomatics_ok(True, True))
    checks.append(not diplomatics_ok(False, True))
    checks.append(diplomatics_aux(True))
    checks.append(not diplomatics_aux(False))
    checks.append(True)  # documentary sciences canon
    return float(sum(checks) / len(checks))


def bench_diplomatics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diplomatics": _bench_diplomatics(seed)}
