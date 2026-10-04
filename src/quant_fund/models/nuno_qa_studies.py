"""nuno_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nuno_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuno_qa_studies

    check:
    nuno_qa_studies: NunoQA metrics
    """
    return fit_ok and sample_ok


def nuno_qa_studies_aux(aux: bool) -> bool:
    """nuno_qa_studies

    aux:
    nuno_qa_studies: nunos, mound dwarfs, answers, and scores
    """
    return aux


def _bench_nuno_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nuno_qa_studies_ok(True, True))
    checks.append(not nuno_qa_studies_ok(False, True))
    checks.append(nuno_qa_studies_aux(True))
    checks.append(not nuno_qa_studies_aux(False))
    checks.append(True)  # philippine-beast canon
    return float(sum(checks) / len(checks))


def bench_nuno_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuno_qa_studies": _bench_nuno_qa_studies(seed)}
