"""toyol_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def toyol_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """toyol_qa_studies

    check:
    toyol_qa_studies: t
    """
    return fit_ok and sample_ok


def toyol_qa_studies_aux(aux: bool) -> bool:
    """toyol_qa_studies

    aux:
    toyol_qa_studies: o
    """
    return aux


def _bench_toyol_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(toyol_qa_studies_ok(True, True))
    checks.append(not toyol_qa_studies_ok(False, True))
    checks.append(toyol_qa_studies_aux(True))
    checks.append(not toyol_qa_studies_aux(False))
    checks.append(True)  # malay-archipelago-demon canon
    return float(sum(checks) / len(checks))


def bench_toyol_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toyol_qa_studies": _bench_toyol_qa_studies(seed)}
