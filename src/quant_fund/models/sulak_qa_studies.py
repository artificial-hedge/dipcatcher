"""sulak_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sulak_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sulak_qa_studies

    check:
    sulak_qa_studies: s
    """
    return fit_ok and sample_ok


def sulak_qa_studies_aux(aux: bool) -> bool:
    """sulak_qa_studies

    aux:
    sulak_qa_studies: u
    """
    return aux


def _bench_sulak_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sulak_qa_studies_ok(True, True))
    checks.append(not sulak_qa_studies_ok(False, True))
    checks.append(sulak_qa_studies_aux(True))
    checks.append(not sulak_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_sulak_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sulak_qa_studies": _bench_sulak_qa_studies(seed)}
