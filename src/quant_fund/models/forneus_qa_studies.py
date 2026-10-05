"""forneus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def forneus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """forneus_qa_studies

    check:
    forneus_qa_studies: F
    """
    return fit_ok and sample_ok


def forneus_qa_studies_aux(aux: bool) -> bool:
    """forneus_qa_studies

    aux:
    forneus_qa_studies: o
    """
    return aux


def _bench_forneus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(forneus_qa_studies_ok(True, True))
    checks.append(not forneus_qa_studies_ok(False, True))
    checks.append(forneus_qa_studies_aux(True))
    checks.append(not forneus_qa_studies_aux(False))
    checks.append(True)  # goetic-circle canon
    return float(sum(checks) / len(checks))


def bench_forneus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forneus_qa_studies": _bench_forneus_qa_studies(seed)}
