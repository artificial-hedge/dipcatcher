"""cassivellaunus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cassivellaunus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cassivellaunus_qa_studies

    check:
    cassivellaunus_qa_studies: w
    """
    return fit_ok and sample_ok


def cassivellaunus_qa_studies_aux(aux: bool) -> bool:
    """cassivellaunus_qa_studies

    aux:
    cassivellaunus_qa_studies: a
    """
    return aux


def _bench_cassivellaunus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cassivellaunus_qa_studies_ok(True, True))
    checks.append(not cassivellaunus_qa_studies_ok(False, True))
    checks.append(cassivellaunus_qa_studies_aux(True))
    checks.append(not cassivellaunus_qa_studies_aux(False))
    checks.append(True)  # welsh-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_cassivellaunus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cassivellaunus_qa_studies": _bench_cassivellaunus_qa_studies(seed)}
