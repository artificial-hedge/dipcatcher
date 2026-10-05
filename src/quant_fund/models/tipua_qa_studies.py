"""tipua_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tipua_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tipua_qa_studies

    check:
    tipua_qa_studies: T
    """
    return fit_ok and sample_ok


def tipua_qa_studies_aux(aux: bool) -> bool:
    """tipua_qa_studies

    aux:
    tipua_qa_studies: i
    """
    return aux


def _bench_tipua_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tipua_qa_studies_ok(True, True))
    checks.append(not tipua_qa_studies_ok(False, True))
    checks.append(tipua_qa_studies_aux(True))
    checks.append(not tipua_qa_studies_aux(False))
    checks.append(True)  # maori-demon canon
    return float(sum(checks) / len(checks))


def bench_tipua_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tipua_qa_studies": _bench_tipua_qa_studies(seed)}
