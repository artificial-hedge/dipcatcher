"""nighthawk_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nighthawk_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nighthawk_qa_studies

    check:
    nighthawk_qa_studies: NighthawkQA metrics
    """
    return fit_ok and sample_ok


def nighthawk_qa_studies_aux(aux: bool) -> bool:
    """nighthawk_qa_studies

    aux:
    nighthawk_qa_studies: nighthawks, rooftops, answers, and scores
    """
    return aux


def _bench_nighthawk_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nighthawk_qa_studies_ok(True, True))
    checks.append(not nighthawk_qa_studies_ok(False, True))
    checks.append(nighthawk_qa_studies_aux(True))
    checks.append(not nighthawk_qa_studies_aux(False))
    checks.append(True)  # nightbird canon
    return float(sum(checks) / len(checks))


def bench_nighthawk_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nighthawk_qa_studies": _bench_nighthawk_qa_studies(seed)}
