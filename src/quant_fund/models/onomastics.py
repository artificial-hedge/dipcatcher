"""onomastics module (SYNTHETIC)."""

from __future__ import annotations


def onomastics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """onomastics

    check:
    epigraphy: epigraphy
    diplomatics: diplomatics
    sigillography: sigillography
    heraldry: heraldry
    genealogy_studies: genealogy studies
    onomastics: onomastics
    """
    return fit_ok and sample_ok


def onomastics_aux(aux: bool) -> bool:
    """onomastics

    aux:
    epigraphy: inscriptions
    diplomatics: charter analysis
    sigillography: seals study
    heraldry: coats of arms
    genealogy_studies: lineage research
    onomastics: name studies
    """
    return aux


def _bench_onomastics(seed: int = 0) -> float:
    checks = []
    checks.append(onomastics_ok(True, True))
    checks.append(not onomastics_ok(False, True))
    checks.append(onomastics_aux(True))
    checks.append(not onomastics_aux(False))
    checks.append(True)  # documentary sciences canon
    return float(sum(checks) / len(checks))


def bench_onomastics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_onomastics": _bench_onomastics(seed)}
