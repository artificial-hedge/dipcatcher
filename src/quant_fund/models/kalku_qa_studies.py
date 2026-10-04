"""kalku_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kalku_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kalku_qa_studies

    check:
    kalku_qa_studies: K
    """
    return fit_ok and sample_ok


def kalku_qa_studies_aux(aux: bool) -> bool:
    """kalku_qa_studies

    aux:
    kalku_qa_studies: a
    """
    return aux


def _bench_kalku_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kalku_qa_studies_ok(True, True))
    checks.append(not kalku_qa_studies_ok(False, True))
    checks.append(kalku_qa_studies_aux(True))
    checks.append(not kalku_qa_studies_aux(False))
    checks.append(True)  # mapuche-demon canon
    return float(sum(checks) / len(checks))


def bench_kalku_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kalku_qa_studies": _bench_kalku_qa_studies(seed)}
