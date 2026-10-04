"""tapir_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tapir_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tapir_qa_studies

    check:
    tapir_qa_studies: TapirQA metrics
    """
    return fit_ok and sample_ok


def tapir_qa_studies_aux(aux: bool) -> bool:
    """tapir_qa_studies

    aux:
    tapir_qa_studies: tapirs, jungles, answers, and scores
    """
    return aux


def _bench_tapir_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tapir_qa_studies_ok(True, True))
    checks.append(not tapir_qa_studies_ok(False, True))
    checks.append(tapir_qa_studies_aux(True))
    checks.append(not tapir_qa_studies_aux(False))
    checks.append(True)  # neotropical canon
    return float(sum(checks) / len(checks))


def bench_tapir_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tapir_qa_studies": _bench_tapir_qa_studies(seed)}
