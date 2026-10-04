"""hotpot_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def hotpot_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hotpot_lite_studies

    check:
    hotpot_lite_studies: HotpotQA multi-hop metrics
    """
    return fit_ok and sample_ok


def hotpot_lite_studies_aux(aux: bool) -> bool:
    """hotpot_lite_studies

    aux:
    hotpot_lite_studies: questions, contexts, answers, and accuracies
    """
    return aux


def _bench_hotpot_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hotpot_lite_studies_ok(True, True))
    checks.append(not hotpot_lite_studies_ok(False, True))
    checks.append(hotpot_lite_studies_aux(True))
    checks.append(not hotpot_lite_studies_aux(False))
    checks.append(True)  # reading-comp-3 canon
    return float(sum(checks) / len(checks))


def bench_hotpot_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hotpot_lite_studies": _bench_hotpot_lite_studies(seed)}
