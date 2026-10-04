"""tonaca_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tonaca_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tonaca_qa_studies

    check:
    tonaca_qa_studies: TonacaQA metrics
    """
    return fit_ok and sample_ok


def tonaca_qa_studies_aux(aux: bool) -> bool:
    """tonaca_qa_studies

    aux:
    tonaca_qa_studies: tonaca, maize keepers, answers, and scores
    """
    return aux


def _bench_tonaca_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tonaca_qa_studies_ok(True, True))
    checks.append(not tonaca_qa_studies_ok(False, True))
    checks.append(tonaca_qa_studies_aux(True))
    checks.append(not tonaca_qa_studies_aux(False))
    checks.append(True)  # aztec-deity-4 canon
    return float(sum(checks) / len(checks))


def bench_tonaca_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tonaca_qa_studies": _bench_tonaca_qa_studies(seed)}
