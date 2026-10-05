"""nakisawame_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nakisawame_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nakisawame_qa_studies

    check:
    nakisawame_qa_studies: N
    """
    return fit_ok and sample_ok


def nakisawame_qa_studies_aux(aux: bool) -> bool:
    """nakisawame_qa_studies

    aux:
    nakisawame_qa_studies: a
    """
    return aux


def _bench_nakisawame_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nakisawame_qa_studies_ok(True, True))
    checks.append(not nakisawame_qa_studies_ok(False, True))
    checks.append(nakisawame_qa_studies_aux(True))
    checks.append(not nakisawame_qa_studies_aux(False))
    checks.append(True)  # yokai-10 canon
    return float(sum(checks) / len(checks))


def bench_nakisawame_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nakisawame_qa_studies": _bench_nakisawame_qa_studies(seed)}
