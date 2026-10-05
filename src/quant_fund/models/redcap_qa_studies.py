"""redcap_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def redcap_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """redcap_qa_studies

    check:
    redcap_qa_studies: R
    """
    return fit_ok and sample_ok


def redcap_qa_studies_aux(aux: bool) -> bool:
    """redcap_qa_studies

    aux:
    redcap_qa_studies: e
    """
    return aux


def _bench_redcap_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(redcap_qa_studies_ok(True, True))
    checks.append(not redcap_qa_studies_ok(False, True))
    checks.append(redcap_qa_studies_aux(True))
    checks.append(not redcap_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_redcap_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_redcap_qa_studies": _bench_redcap_qa_studies(seed)}
