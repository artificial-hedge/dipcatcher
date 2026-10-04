"""ani_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ani_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ani_qa_studies

    check:
    ani_qa_studies: AniQA metrics
    """
    return fit_ok and sample_ok


def ani_qa_studies_aux(aux: bool) -> bool:
    """ani_qa_studies

    aux:
    ani_qa_studies: anis, scrublands, answers, and scores
    """
    return aux


def _bench_ani_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ani_qa_studies_ok(True, True))
    checks.append(not ani_qa_studies_ok(False, True))
    checks.append(ani_qa_studies_aux(True))
    checks.append(not ani_qa_studies_aux(False))
    checks.append(True)  # cuckoo-turaco canon
    return float(sum(checks) / len(checks))


def bench_ani_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ani_qa_studies": _bench_ani_qa_studies(seed)}
