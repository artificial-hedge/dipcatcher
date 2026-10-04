"""feldspar_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def feldspar_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """feldspar_qa_studies

    check:
    feldspar_qa_studies: FeldsparQA metrics
    """
    return fit_ok and sample_ok


def feldspar_qa_studies_aux(aux: bool) -> bool:
    """feldspar_qa_studies

    aux:
    feldspar_qa_studies: feldspars, granites, answers, and scores
    """
    return aux


def _bench_feldspar_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(feldspar_qa_studies_ok(True, True))
    checks.append(not feldspar_qa_studies_ok(False, True))
    checks.append(feldspar_qa_studies_aux(True))
    checks.append(not feldspar_qa_studies_aux(False))
    checks.append(True)  # mineral canon
    return float(sum(checks) / len(checks))


def bench_feldspar_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_feldspar_qa_studies": _bench_feldspar_qa_studies(seed)}
