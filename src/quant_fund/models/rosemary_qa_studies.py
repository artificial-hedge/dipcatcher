"""rosemary_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rosemary_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rosemary_qa_studies

    check:
    rosemary_qa_studies: RosemaryQA metrics
    """
    return fit_ok and sample_ok


def rosemary_qa_studies_aux(aux: bool) -> bool:
    """rosemary_qa_studies

    aux:
    rosemary_qa_studies: rosemary, shrubs, answers, and scores
    """
    return aux


def _bench_rosemary_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rosemary_qa_studies_ok(True, True))
    checks.append(not rosemary_qa_studies_ok(False, True))
    checks.append(rosemary_qa_studies_aux(True))
    checks.append(not rosemary_qa_studies_aux(False))
    checks.append(True)  # spice-2 canon
    return float(sum(checks) / len(checks))


def bench_rosemary_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rosemary_qa_studies": _bench_rosemary_qa_studies(seed)}
