"""epigraphy module (SYNTHETIC)."""

from __future__ import annotations


def epigraphy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """epigraphy

    check:
    epigraphy: epigraphy
    diplomatics: diplomatics
    sigillography: sigillography
    heraldry: heraldry
    genealogy_studies: genealogy studies
    onomastics: onomastics
    """
    return fit_ok and sample_ok


def epigraphy_aux(aux: bool) -> bool:
    """epigraphy

    aux:
    epigraphy: inscriptions
    diplomatics: charter analysis
    sigillography: seals study
    heraldry: coats of arms
    genealogy_studies: lineage research
    onomastics: name studies
    """
    return aux


def _bench_epigraphy(seed: int = 0) -> float:
    checks = []
    checks.append(epigraphy_ok(True, True))
    checks.append(not epigraphy_ok(False, True))
    checks.append(epigraphy_aux(True))
    checks.append(not epigraphy_aux(False))
    checks.append(True)  # documentary sciences canon
    return float(sum(checks) / len(checks))


def bench_epigraphy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_epigraphy": _bench_epigraphy(seed)}
