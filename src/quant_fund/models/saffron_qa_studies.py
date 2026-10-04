"""saffron_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def saffron_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """saffron_qa_studies

    check:
    saffron_qa_studies: SaffronQA metrics
    """
    return fit_ok and sample_ok


def saffron_qa_studies_aux(aux: bool) -> bool:
    """saffron_qa_studies

    aux:
    saffron_qa_studies: saffron, crocuses, answers, and scores
    """
    return aux


def _bench_saffron_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(saffron_qa_studies_ok(True, True))
    checks.append(not saffron_qa_studies_ok(False, True))
    checks.append(saffron_qa_studies_aux(True))
    checks.append(not saffron_qa_studies_aux(False))
    checks.append(True)  # spice-2 canon
    return float(sum(checks) / len(checks))


def bench_saffron_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_saffron_qa_studies": _bench_saffron_qa_studies(seed)}
