"""dentistry_2 module (SYNTHETIC)."""

from __future__ import annotations


def dentistry_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dentistry_2

    check:
    optometry: optometry
    dentistry_2: dentistry
    podiatry: podiatry
    dietetics: dietetics
    physiotherapy: physiotherapy
    occupational_therapy: occupational therapy
    """
    return fit_ok and sample_ok


def dentistry_2_aux(aux: bool) -> bool:
    """dentistry_2

    aux:
    optometry: vision correction
    dentistry_2: oral health
    podiatry: foot and ankle
    dietetics: nutrition and diet
    physiotherapy: movement and mobility
    occupational_therapy: daily function
    """
    return aux


def _bench_dentistry_2(seed: int = 0) -> float:
    checks = []
    checks.append(dentistry_2_ok(True, True))
    checks.append(not dentistry_2_ok(False, True))
    checks.append(dentistry_2_aux(True))
    checks.append(not dentistry_2_aux(False))
    checks.append(True)  # medicine-6 canon
    return float(sum(checks) / len(checks))


def bench_dentistry_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dentistry_2": _bench_dentistry_2(seed)}
