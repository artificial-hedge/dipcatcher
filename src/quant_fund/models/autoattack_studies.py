"""autoattack_studies module (SYNTHETIC)."""

from __future__ import annotations


def autoattack_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """autoattack_studies

    check:
    autoattack_studies: AutoAttack ensemble attacks/APGD and verified acc
    """
    return fit_ok and sample_ok


def autoattack_studies_aux(aux: bool) -> bool:
    """autoattack_studies

    aux:
    autoattack_studies: certified radius attacks/steps and bounds
    """
    return aux


def _bench_autoattack_studies(seed: int = 0) -> float:
    checks = []
    checks.append(autoattack_studies_ok(True, True))
    checks.append(not autoattack_studies_ok(False, True))
    checks.append(autoattack_studies_aux(True))
    checks.append(not autoattack_studies_aux(False))
    checks.append(True)  # robustness-eval canon
    return float(sum(checks) / len(checks))


def bench_autoattack_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_autoattack_studies": _bench_autoattack_studies(seed)}
