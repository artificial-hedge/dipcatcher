"""rhino_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rhino_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rhino_qa_studies

    check:
    rhino_qa_studies: RhinoQA metrics
    """
    return fit_ok and sample_ok


def rhino_qa_studies_aux(aux: bool) -> bool:
    """rhino_qa_studies

    aux:
    rhino_qa_studies: rhinos, thorn scrub, answers, and scores
    """
    return aux


def _bench_rhino_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rhino_qa_studies_ok(True, True))
    checks.append(not rhino_qa_studies_ok(False, True))
    checks.append(rhino_qa_studies_aux(True))
    checks.append(not rhino_qa_studies_aux(False))
    checks.append(True)  # savanna-herd canon
    return float(sum(checks) / len(checks))


def bench_rhino_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rhino_qa_studies": _bench_rhino_qa_studies(seed)}
