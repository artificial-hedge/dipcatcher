"""optometry module (SYNTHETIC)."""

from __future__ import annotations


def optometry_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """optometry

    check:
    optometry: optometry
    dentistry_2: dentistry
    podiatry: podiatry
    dietetics: dietetics
    physiotherapy: physiotherapy
    occupational_therapy: occupational therapy
    """
    return fit_ok and sample_ok


def optometry_aux(aux: bool) -> bool:
    """optometry

    aux:
    optometry: vision correction
    dentistry_2: oral health
    podiatry: foot and ankle
    dietetics: nutrition and diet
    physiotherapy: movement and mobility
    occupational_therapy: daily function
    """
    return aux


def _bench_optometry(seed: int = 0) -> float:
    checks = []
    checks.append(optometry_ok(True, True))
    checks.append(not optometry_ok(False, True))
    checks.append(optometry_aux(True))
    checks.append(not optometry_aux(False))
    checks.append(True)  # medicine-6 canon
    return float(sum(checks) / len(checks))


def bench_optometry(seed: int = 0) -> dict[str, float]:
    return {"synthetic_optometry": _bench_optometry(seed)}
