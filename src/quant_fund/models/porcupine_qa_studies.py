"""porcupine_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def porcupine_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """porcupine_qa_studies

    check:
    porcupine_qa_studies: PorcupineQA metrics
    """
    return fit_ok and sample_ok


def porcupine_qa_studies_aux(aux: bool) -> bool:
    """porcupine_qa_studies

    aux:
    porcupine_qa_studies: porcupines, root hollows, answers, and scores
    """
    return aux


def _bench_porcupine_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(porcupine_qa_studies_ok(True, True))
    checks.append(not porcupine_qa_studies_ok(False, True))
    checks.append(porcupine_qa_studies_aux(True))
    checks.append(not porcupine_qa_studies_aux(False))
    checks.append(True)  # small-mammal-2 canon
    return float(sum(checks) / len(checks))


def bench_porcupine_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_porcupine_qa_studies": _bench_porcupine_qa_studies(seed)}
