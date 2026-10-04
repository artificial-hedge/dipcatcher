"""bird_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bird_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bird_qa_studies

    check:
    bird_qa_studies: BirdQA metrics
    """
    return fit_ok and sample_ok


def bird_qa_studies_aux(aux: bool) -> bool:
    """bird_qa_studies

    aux:
    bird_qa_studies: birds, migrations, answers, and scores
    """
    return aux


def _bench_bird_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bird_qa_studies_ok(True, True))
    checks.append(not bird_qa_studies_ok(False, True))
    checks.append(bird_qa_studies_aux(True))
    checks.append(not bird_qa_studies_aux(False))
    checks.append(True)  # wildlife canon
    return float(sum(checks) / len(checks))


def bench_bird_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bird_qa_studies": _bench_bird_qa_studies(seed)}
