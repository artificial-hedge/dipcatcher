"""pontianak_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pontianak_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pontianak_qa_studies

    check:
    pontianak_qa_studies: p
    """
    return fit_ok and sample_ok


def pontianak_qa_studies_aux(aux: bool) -> bool:
    """pontianak_qa_studies

    aux:
    pontianak_qa_studies: o
    """
    return aux


def _bench_pontianak_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pontianak_qa_studies_ok(True, True))
    checks.append(not pontianak_qa_studies_ok(False, True))
    checks.append(pontianak_qa_studies_aux(True))
    checks.append(not pontianak_qa_studies_aux(False))
    checks.append(True)  # malay-archipelago-demon canon
    return float(sum(checks) / len(checks))


def bench_pontianak_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pontianak_qa_studies": _bench_pontianak_qa_studies(seed)}
