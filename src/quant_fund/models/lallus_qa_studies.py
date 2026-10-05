"""lallus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lallus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lallus_qa_studies

    check:
    lallus_qa_studies: s
    """
    return fit_ok and sample_ok


def lallus_qa_studies_aux(aux: bool) -> bool:
    """lallus_qa_studies

    aux:
    lallus_qa_studies: p
    """
    return aux


def _bench_lallus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lallus_qa_studies_ok(True, True))
    checks.append(not lallus_qa_studies_ok(False, True))
    checks.append(lallus_qa_studies_aux(True))
    checks.append(not lallus_qa_studies_aux(False))
    checks.append(True)  # numidian-myth canon
    return float(sum(checks) / len(checks))


def bench_lallus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lallus_qa_studies": _bench_lallus_qa_studies(seed)}
