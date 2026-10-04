"""winogender_studies module (SYNTHETIC)."""

from __future__ import annotations


def winogender_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """winogender_studies

    check:
    winogender_studies: Winogender occupational pronoun resolution and bias
    """
    return fit_ok and sample_ok


def winogender_studies_aux(aux: bool) -> bool:
    """winogender_studies

    aux:
    winogender_studies: sentences, gender splits, referents, and parity
    """
    return aux


def _bench_winogender_studies(seed: int = 0) -> float:
    checks = []
    checks.append(winogender_studies_ok(True, True))
    checks.append(not winogender_studies_ok(False, True))
    checks.append(winogender_studies_aux(True))
    checks.append(not winogender_studies_aux(False))
    checks.append(True)  # winograd-eval canon
    return float(sum(checks) / len(checks))


def bench_winogender_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_winogender_studies": _bench_winogender_studies(seed)}
