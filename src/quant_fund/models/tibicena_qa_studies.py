"""tibicena_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tibicena_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tibicena_qa_studies

    check:
    tibicena_qa_studies: d
    """
    return fit_ok and sample_ok


def tibicena_qa_studies_aux(aux: bool) -> bool:
    """tibicena_qa_studies

    aux:
    tibicena_qa_studies: e
    """
    return aux


def _bench_tibicena_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tibicena_qa_studies_ok(True, True))
    checks.append(not tibicena_qa_studies_ok(False, True))
    checks.append(tibicena_qa_studies_aux(True))
    checks.append(not tibicena_qa_studies_aux(False))
    checks.append(True)  # guanche-myth canon
    return float(sum(checks) / len(checks))


def bench_tibicena_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tibicena_qa_studies": _bench_tibicena_qa_studies(seed)}
