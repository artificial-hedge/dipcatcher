"""morveren_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def morveren_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """morveren_qa_studies

    check:
    morveren_qa_studies: s
    """
    return fit_ok and sample_ok


def morveren_qa_studies_aux(aux: bool) -> bool:
    """morveren_qa_studies

    aux:
    morveren_qa_studies: e
    """
    return aux


def _bench_morveren_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(morveren_qa_studies_ok(True, True))
    checks.append(not morveren_qa_studies_ok(False, True))
    checks.append(morveren_qa_studies_aux(True))
    checks.append(not morveren_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-6 canon
    return float(sum(checks) / len(checks))


def bench_morveren_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morveren_qa_studies": _bench_morveren_qa_studies(seed)}
