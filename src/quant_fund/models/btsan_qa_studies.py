"""btsan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def btsan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """btsan_qa_studies

    check:
    btsan_qa_studies: B
    """
    return fit_ok and sample_ok


def btsan_qa_studies_aux(aux: bool) -> bool:
    """btsan_qa_studies

    aux:
    btsan_qa_studies: t
    """
    return aux


def _bench_btsan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(btsan_qa_studies_ok(True, True))
    checks.append(not btsan_qa_studies_ok(False, True))
    checks.append(btsan_qa_studies_aux(True))
    checks.append(not btsan_qa_studies_aux(False))
    checks.append(True)  # tibetan-demon canon
    return float(sum(checks) / len(checks))


def bench_btsan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_btsan_qa_studies": _bench_btsan_qa_studies(seed)}
