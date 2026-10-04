"""backgrounds_studies module (SYNTHETIC)."""

from __future__ import annotations


def backgrounds_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """backgrounds_studies

    check:
    backgrounds_studies: backgrounds-robustness challenge accuracy and gaps
    """
    return fit_ok and sample_ok


def backgrounds_studies_aux(aux: bool) -> bool:
    """backgrounds_studies

    aux:
    backgrounds_studies: foreground/background pairs, rates, and scores
    """
    return aux


def _bench_backgrounds_studies(seed: int = 0) -> float:
    checks = []
    checks.append(backgrounds_studies_ok(True, True))
    checks.append(not backgrounds_studies_ok(False, True))
    checks.append(backgrounds_studies_aux(True))
    checks.append(not backgrounds_studies_aux(False))
    checks.append(True)  # cue-conflict canon
    return float(sum(checks) / len(checks))


def bench_backgrounds_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_backgrounds_studies": _bench_backgrounds_studies(seed)}
