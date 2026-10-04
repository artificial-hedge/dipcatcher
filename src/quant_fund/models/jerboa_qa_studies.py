"""jerboa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jerboa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jerboa_qa_studies

    check:
    jerboa_qa_studies: JerboaQA metrics
    """
    return fit_ok and sample_ok


def jerboa_qa_studies_aux(aux: bool) -> bool:
    """jerboa_qa_studies

    aux:
    jerboa_qa_studies: jerboas, sandy dunes, answers, and scores
    """
    return aux


def _bench_jerboa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jerboa_qa_studies_ok(True, True))
    checks.append(not jerboa_qa_studies_ok(False, True))
    checks.append(jerboa_qa_studies_aux(True))
    checks.append(not jerboa_qa_studies_aux(False))
    checks.append(True)  # desert canon
    return float(sum(checks) / len(checks))


def bench_jerboa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jerboa_qa_studies": _bench_jerboa_qa_studies(seed)}
