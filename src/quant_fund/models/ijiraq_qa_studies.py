"""ijiraq_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ijiraq_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ijiraq_qa_studies

    check:
    ijiraq_qa_studies: I
    """
    return fit_ok and sample_ok


def ijiraq_qa_studies_aux(aux: bool) -> bool:
    """ijiraq_qa_studies

    aux:
    ijiraq_qa_studies: j
    """
    return aux


def _bench_ijiraq_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ijiraq_qa_studies_ok(True, True))
    checks.append(not ijiraq_qa_studies_ok(False, True))
    checks.append(ijiraq_qa_studies_aux(True))
    checks.append(not ijiraq_qa_studies_aux(False))
    checks.append(True)  # inuit-demon canon
    return float(sum(checks) / len(checks))


def bench_ijiraq_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ijiraq_qa_studies": _bench_ijiraq_qa_studies(seed)}
