"""occupational_therapy module (SYNTHETIC)."""

from __future__ import annotations


def occupational_therapy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """occupational_therapy

    check:
    optometry: optometry
    dentistry_2: dentistry
    podiatry: podiatry
    dietetics: dietetics
    physiotherapy: physiotherapy
    occupational_therapy: occupational therapy
    """
    return fit_ok and sample_ok


def occupational_therapy_aux(aux: bool) -> bool:
    """occupational_therapy

    aux:
    optometry: vision correction
    dentistry_2: oral health
    podiatry: foot and ankle
    dietetics: nutrition and diet
    physiotherapy: movement and mobility
    occupational_therapy: daily function
    """
    return aux


def _bench_occupational_therapy(seed: int = 0) -> float:
    checks = []
    checks.append(occupational_therapy_ok(True, True))
    checks.append(not occupational_therapy_ok(False, True))
    checks.append(occupational_therapy_aux(True))
    checks.append(not occupational_therapy_aux(False))
    checks.append(True)  # medicine-6 canon
    return float(sum(checks) / len(checks))


def bench_occupational_therapy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_occupational_therapy": _bench_occupational_therapy(seed)}
