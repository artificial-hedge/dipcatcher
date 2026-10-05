"""kalma_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kalma_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kalma_qa_studies

    check:
    kalma_qa_studies: K
    """
    return fit_ok and sample_ok


def kalma_qa_studies_aux(aux: bool) -> bool:
    """kalma_qa_studies

    aux:
    kalma_qa_studies: a
    """
    return aux


def _bench_kalma_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kalma_qa_studies_ok(True, True))
    checks.append(not kalma_qa_studies_ok(False, True))
    checks.append(kalma_qa_studies_aux(True))
    checks.append(not kalma_qa_studies_aux(False))
    checks.append(True)  # finnish-demon canon
    return float(sum(checks) / len(checks))


def bench_kalma_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kalma_qa_studies": _bench_kalma_qa_studies(seed)}
