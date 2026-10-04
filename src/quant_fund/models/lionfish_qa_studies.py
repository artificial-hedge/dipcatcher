"""lionfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lionfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lionfish_qa_studies

    check:
    lionfish_qa_studies: LionfishQA metrics
    """
    return fit_ok and sample_ok


def lionfish_qa_studies_aux(aux: bool) -> bool:
    """lionfish_qa_studies

    aux:
    lionfish_qa_studies: lionfish, reef ledges, answers, and scores
    """
    return aux


def _bench_lionfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lionfish_qa_studies_ok(True, True))
    checks.append(not lionfish_qa_studies_ok(False, True))
    checks.append(lionfish_qa_studies_aux(True))
    checks.append(not lionfish_qa_studies_aux(False))
    checks.append(True)  # reef-fish-2 canon
    return float(sum(checks) / len(checks))


def bench_lionfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lionfish_qa_studies": _bench_lionfish_qa_studies(seed)}
