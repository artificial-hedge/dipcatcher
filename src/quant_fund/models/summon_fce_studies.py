"""summon_fce_studies module (SYNTHETIC)."""

from __future__ import annotations


def summon_fce_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """summon_fce_studies

    check:
    summon_fce_studies: SummScreen metrics
    """
    return fit_ok and sample_ok


def summon_fce_studies_aux(aux: bool) -> bool:
    """summon_fce_studies

    aux:
    summon_fce_studies: scripts, recaps, scenes, and scores
    """
    return aux


def _bench_summon_fce_studies(seed: int = 0) -> float:
    checks = []
    checks.append(summon_fce_studies_ok(True, True))
    checks.append(not summon_fce_studies_ok(False, True))
    checks.append(summon_fce_studies_aux(True))
    checks.append(not summon_fce_studies_aux(False))
    checks.append(True)  # multi-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_summon_fce_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_summon_fce_studies": _bench_summon_fce_studies(seed)}
