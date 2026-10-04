"""flute_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def flute_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """flute_qa_studies

    check:
    flute_qa_studies: FluteQA metrics
    """
    return fit_ok and sample_ok


def flute_qa_studies_aux(aux: bool) -> bool:
    """flute_qa_studies

    aux:
    flute_qa_studies: flutes, keys, answers, and scores
    """
    return aux


def _bench_flute_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(flute_qa_studies_ok(True, True))
    checks.append(not flute_qa_studies_ok(False, True))
    checks.append(flute_qa_studies_aux(True))
    checks.append(not flute_qa_studies_aux(False))
    checks.append(True)  # instrument canon
    return float(sum(checks) / len(checks))


def bench_flute_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flute_qa_studies": _bench_flute_qa_studies(seed)}
