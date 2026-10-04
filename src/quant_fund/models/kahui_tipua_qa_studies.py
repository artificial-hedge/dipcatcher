"""kahui_tipua_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kahui_tipua_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kahui_tipua_qa_studies

    check:
    kahui_tipua_qa_studies: K
    """
    return fit_ok and sample_ok


def kahui_tipua_qa_studies_aux(aux: bool) -> bool:
    """kahui_tipua_qa_studies

    aux:
    kahui_tipua_qa_studies: a
    """
    return aux


def _bench_kahui_tipua_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kahui_tipua_qa_studies_ok(True, True))
    checks.append(not kahui_tipua_qa_studies_ok(False, True))
    checks.append(kahui_tipua_qa_studies_aux(True))
    checks.append(not kahui_tipua_qa_studies_aux(False))
    checks.append(True)  # maori-demon canon
    return float(sum(checks) / len(checks))


def bench_kahui_tipua_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kahui_tipua_qa_studies": _bench_kahui_tipua_qa_studies(seed)}
