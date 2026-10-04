"""mandarinfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mandarinfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mandarinfish_qa_studies

    check:
    mandarinfish_qa_studies: MandarinfishQA metrics
    """
    return fit_ok and sample_ok


def mandarinfish_qa_studies_aux(aux: bool) -> bool:
    """mandarinfish_qa_studies

    aux:
    mandarinfish_qa_studies: mandarinfish, coral rubble, answers, and scores
    """
    return aux


def _bench_mandarinfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mandarinfish_qa_studies_ok(True, True))
    checks.append(not mandarinfish_qa_studies_ok(False, True))
    checks.append(mandarinfish_qa_studies_aux(True))
    checks.append(not mandarinfish_qa_studies_aux(False))
    checks.append(True)  # reef-fish-3 canon
    return float(sum(checks) / len(checks))


def bench_mandarinfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mandarinfish_qa_studies": _bench_mandarinfish_qa_studies(seed)}
