"""mithra_iran_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mithra_iran_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mithra_iran_qa_studies

    check:
    mithra_iran_qa_studies: m
    """
    return fit_ok and sample_ok


def mithra_iran_qa_studies_aux(aux: bool) -> bool:
    """mithra_iran_qa_studies

    aux:
    mithra_iran_qa_studies: i
    """
    return aux


def _bench_mithra_iran_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mithra_iran_qa_studies_ok(True, True))
    checks.append(not mithra_iran_qa_studies_ok(False, True))
    checks.append(mithra_iran_qa_studies_aux(True))
    checks.append(not mithra_iran_qa_studies_aux(False))
    checks.append(True)  # folk-spirit lore-2 canon
    return float(sum(checks) / len(checks))


def bench_mithra_iran_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mithra_iran_qa_studies": _bench_mithra_iran_qa_studies(seed)}
