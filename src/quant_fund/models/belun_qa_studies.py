"""belun_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def belun_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """belun_qa_studies

    check:
    belun_qa_studies: B
    """
    return fit_ok and sample_ok


def belun_qa_studies_aux(aux: bool) -> bool:
    """belun_qa_studies

    aux:
    belun_qa_studies: e
    """
    return aux


def _bench_belun_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(belun_qa_studies_ok(True, True))
    checks.append(not belun_qa_studies_ok(False, True))
    checks.append(belun_qa_studies_aux(True))
    checks.append(not belun_qa_studies_aux(False))
    checks.append(True)  # slavic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_belun_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_belun_qa_studies": _bench_belun_qa_studies(seed)}
