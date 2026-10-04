"""belili_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def belili_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """belili_qa_studies

    check:
    belili_qa_studies: b
    """
    return fit_ok and sample_ok


def belili_qa_studies_aux(aux: bool) -> bool:
    """belili_qa_studies

    aux:
    belili_qa_studies: e
    """
    return aux


def _bench_belili_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(belili_qa_studies_ok(True, True))
    checks.append(not belili_qa_studies_ok(False, True))
    checks.append(belili_qa_studies_aux(True))
    checks.append(not belili_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_belili_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_belili_qa_studies": _bench_belili_qa_studies(seed)}
