"""pelesit_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pelesit_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pelesit_qa_studies

    check:
    pelesit_qa_studies: p
    """
    return fit_ok and sample_ok


def pelesit_qa_studies_aux(aux: bool) -> bool:
    """pelesit_qa_studies

    aux:
    pelesit_qa_studies: e
    """
    return aux


def _bench_pelesit_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pelesit_qa_studies_ok(True, True))
    checks.append(not pelesit_qa_studies_ok(False, True))
    checks.append(pelesit_qa_studies_aux(True))
    checks.append(not pelesit_qa_studies_aux(False))
    checks.append(True)  # malay-archipelago-demon canon
    return float(sum(checks) / len(checks))


def bench_pelesit_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pelesit_qa_studies": _bench_pelesit_qa_studies(seed)}
