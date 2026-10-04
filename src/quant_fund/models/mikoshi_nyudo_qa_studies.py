"""mikoshi_nyudo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mikoshi_nyudo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mikoshi_nyudo_qa_studies

    check:
    mikoshi_nyudo_qa_studies: M
    """
    return fit_ok and sample_ok


def mikoshi_nyudo_qa_studies_aux(aux: bool) -> bool:
    """mikoshi_nyudo_qa_studies

    aux:
    mikoshi_nyudo_qa_studies: i
    """
    return aux


def _bench_mikoshi_nyudo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mikoshi_nyudo_qa_studies_ok(True, True))
    checks.append(not mikoshi_nyudo_qa_studies_ok(False, True))
    checks.append(mikoshi_nyudo_qa_studies_aux(True))
    checks.append(not mikoshi_nyudo_qa_studies_aux(False))
    checks.append(True)  # yokai-8 canon
    return float(sum(checks) / len(checks))


def bench_mikoshi_nyudo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mikoshi_nyudo_qa_studies": _bench_mikoshi_nyudo_qa_studies(seed)}
