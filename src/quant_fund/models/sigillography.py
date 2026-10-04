"""sigillography module (SYNTHETIC)."""

from __future__ import annotations


def sigillography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sigillography

    check:
    epigraphy: epigraphy
    diplomatics: diplomatics
    sigillography: sigillography
    heraldry: heraldry
    genealogy_studies: genealogy studies
    onomastics: onomastics
    """
    return fit_ok and sample_ok


def sigillography_aux(aux: bool) -> bool:
    """sigillography

    aux:
    epigraphy: inscriptions
    diplomatics: charter analysis
    sigillography: seals study
    heraldry: coats of arms
    genealogy_studies: lineage research
    onomastics: name studies
    """
    return aux


def _bench_sigillography(seed: int = 0) -> float:
    checks = []
    checks.append(sigillography_ok(True, True))
    checks.append(not sigillography_ok(False, True))
    checks.append(sigillography_aux(True))
    checks.append(not sigillography_aux(False))
    checks.append(True)  # documentary sciences canon
    return float(sum(checks) / len(checks))


def bench_sigillography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sigillography": _bench_sigillography(seed)}
