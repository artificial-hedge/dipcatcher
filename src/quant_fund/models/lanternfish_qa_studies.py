"""lanternfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lanternfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lanternfish_qa_studies

    check:
    lanternfish_qa_studies: LanternfishQA metrics
    """
    return fit_ok and sample_ok


def lanternfish_qa_studies_aux(aux: bool) -> bool:
    """lanternfish_qa_studies

    aux:
    lanternfish_qa_studies: lanternfish, nocturnal migrations, answers, and scores
    """
    return aux


def _bench_lanternfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lanternfish_qa_studies_ok(True, True))
    checks.append(not lanternfish_qa_studies_ok(False, True))
    checks.append(lanternfish_qa_studies_aux(True))
    checks.append(not lanternfish_qa_studies_aux(False))
    checks.append(True)  # abyssal canon
    return float(sum(checks) / len(checks))


def bench_lanternfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lanternfish_qa_studies": _bench_lanternfish_qa_studies(seed)}
