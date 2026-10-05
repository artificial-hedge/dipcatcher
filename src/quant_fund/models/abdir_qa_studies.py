"""abdir_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def abdir_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """abdir_qa_studies

    check:
    abdir_qa_studies: s
    """
    return fit_ok and sample_ok


def abdir_qa_studies_aux(aux: bool) -> bool:
    """abdir_qa_studies

    aux:
    abdir_qa_studies: e
    """
    return aux


def _bench_abdir_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(abdir_qa_studies_ok(True, True))
    checks.append(not abdir_qa_studies_ok(False, True))
    checks.append(abdir_qa_studies_aux(True))
    checks.append(not abdir_qa_studies_aux(False))
    checks.append(True)  # punic-4 canon
    return float(sum(checks) / len(checks))


def bench_abdir_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abdir_qa_studies": _bench_abdir_qa_studies(seed)}
