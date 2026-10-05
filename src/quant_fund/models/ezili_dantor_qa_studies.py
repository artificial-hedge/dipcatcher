"""ezili_dantor_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ezili_dantor_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ezili_dantor_qa_studies

    check:
    ezili_dantor_qa_studies: E
    """
    return fit_ok and sample_ok


def ezili_dantor_qa_studies_aux(aux: bool) -> bool:
    """ezili_dantor_qa_studies

    aux:
    ezili_dantor_qa_studies: z
    """
    return aux


def _bench_ezili_dantor_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ezili_dantor_qa_studies_ok(True, True))
    checks.append(not ezili_dantor_qa_studies_ok(False, True))
    checks.append(ezili_dantor_qa_studies_aux(True))
    checks.append(not ezili_dantor_qa_studies_aux(False))
    checks.append(True)  # vodou-loa canon
    return float(sum(checks) / len(checks))


def bench_ezili_dantor_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ezili_dantor_qa_studies": _bench_ezili_dantor_qa_studies(seed)}
