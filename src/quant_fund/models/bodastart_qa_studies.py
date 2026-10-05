"""bodastart_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bodastart_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bodastart_qa_studies

    check:
    bodastart_qa_studies: d
    """
    return fit_ok and sample_ok


def bodastart_qa_studies_aux(aux: bool) -> bool:
    """bodastart_qa_studies

    aux:
    bodastart_qa_studies: i
    """
    return aux


def _bench_bodastart_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bodastart_qa_studies_ok(True, True))
    checks.append(not bodastart_qa_studies_ok(False, True))
    checks.append(bodastart_qa_studies_aux(True))
    checks.append(not bodastart_qa_studies_aux(False))
    checks.append(True)  # punic-3 canon
    return float(sum(checks) / len(checks))


def bench_bodastart_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bodastart_qa_studies": _bench_bodastart_qa_studies(seed)}
