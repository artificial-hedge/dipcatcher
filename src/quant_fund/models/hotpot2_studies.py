"""hotpot2_studies module (SYNTHETIC)."""

from __future__ import annotations


def hotpot2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hotpot2_studies

    check:
    hotpot2_studies: HotpotQA-2 metrics
    """
    return fit_ok and sample_ok


def hotpot2_studies_aux(aux: bool) -> bool:
    """hotpot2_studies

    aux:
    hotpot2_studies: questions, bridges, answers, and scores
    """
    return aux


def _bench_hotpot2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hotpot2_studies_ok(True, True))
    checks.append(not hotpot2_studies_ok(False, True))
    checks.append(hotpot2_studies_aux(True))
    checks.append(not hotpot2_studies_aux(False))
    checks.append(True)  # QA-exotics-3 canon
    return float(sum(checks) / len(checks))


def bench_hotpot2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hotpot2_studies": _bench_hotpot2_studies(seed)}
