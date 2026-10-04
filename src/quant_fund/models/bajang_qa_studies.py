"""bajang_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bajang_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bajang_qa_studies

    check:
    bajang_qa_studies: b
    """
    return fit_ok and sample_ok


def bajang_qa_studies_aux(aux: bool) -> bool:
    """bajang_qa_studies

    aux:
    bajang_qa_studies: a
    """
    return aux


def _bench_bajang_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bajang_qa_studies_ok(True, True))
    checks.append(not bajang_qa_studies_ok(False, True))
    checks.append(bajang_qa_studies_aux(True))
    checks.append(not bajang_qa_studies_aux(False))
    checks.append(True)  # malay-archipelago-demon canon
    return float(sum(checks) / len(checks))


def bench_bajang_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bajang_qa_studies": _bench_bajang_qa_studies(seed)}
