"""caddisfly_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def caddisfly_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """caddisfly_qa_studies

    check:
    caddisfly_qa_studies: CaddisflyQA metrics
    """
    return fit_ok and sample_ok


def caddisfly_qa_studies_aux(aux: bool) -> bool:
    """caddisfly_qa_studies

    aux:
    caddisfly_qa_studies: caddisflies, streams, answers, and scores
    """
    return aux


def _bench_caddisfly_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(caddisfly_qa_studies_ok(True, True))
    checks.append(not caddisfly_qa_studies_ok(False, True))
    checks.append(caddisfly_qa_studies_aux(True))
    checks.append(not caddisfly_qa_studies_aux(False))
    checks.append(True)  # invertebrate-2 canon
    return float(sum(checks) / len(checks))


def bench_caddisfly_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_caddisfly_qa_studies": _bench_caddisfly_qa_studies(seed)}
