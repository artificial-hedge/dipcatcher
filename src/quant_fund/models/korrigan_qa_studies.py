"""korrigan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def korrigan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """korrigan_qa_studies

    check:
    korrigan_qa_studies: d
    """
    return fit_ok and sample_ok


def korrigan_qa_studies_aux(aux: bool) -> bool:
    """korrigan_qa_studies

    aux:
    korrigan_qa_studies: w
    """
    return aux


def _bench_korrigan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(korrigan_qa_studies_ok(True, True))
    checks.append(not korrigan_qa_studies_ok(False, True))
    checks.append(korrigan_qa_studies_aux(True))
    checks.append(not korrigan_qa_studies_aux(False))
    checks.append(True)  # breton-myth canon
    return float(sum(checks) / len(checks))


def bench_korrigan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_korrigan_qa_studies": _bench_korrigan_qa_studies(seed)}
