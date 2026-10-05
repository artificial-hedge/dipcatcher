"""changxi2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def changxi2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """changxi2_qa_studies

    check:
    changxi2_qa_studies: Changxi2QA metrics
    """
    return fit_ok and sample_ok


def changxi2_qa_studies_aux(aux: bool) -> bool:
    """changxi2_qa_studies

    aux:
    changxi2_qa_studies: changxi2, moon mothers, answers, and scores
    """
    return aux


def _bench_changxi2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(changxi2_qa_studies_ok(True, True))
    checks.append(not changxi2_qa_studies_ok(False, True))
    checks.append(changxi2_qa_studies_aux(True))
    checks.append(not changxi2_qa_studies_aux(False))
    checks.append(True)  # chinese-myth-6 canon
    return float(sum(checks) / len(checks))


def bench_changxi2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_changxi2_qa_studies": _bench_changxi2_qa_studies(seed)}
