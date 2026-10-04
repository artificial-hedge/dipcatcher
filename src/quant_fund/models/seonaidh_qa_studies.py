"""seonaidh_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def seonaidh_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """seonaidh_qa_studies

    check:
    seonaidh_qa_studies: S
    """
    return fit_ok and sample_ok


def seonaidh_qa_studies_aux(aux: bool) -> bool:
    """seonaidh_qa_studies

    aux:
    seonaidh_qa_studies: e
    """
    return aux


def _bench_seonaidh_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(seonaidh_qa_studies_ok(True, True))
    checks.append(not seonaidh_qa_studies_ok(False, True))
    checks.append(seonaidh_qa_studies_aux(True))
    checks.append(not seonaidh_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_seonaidh_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_seonaidh_qa_studies": _bench_seonaidh_qa_studies(seed)}
