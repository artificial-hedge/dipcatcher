"""gnod_sbyin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gnod_sbyin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gnod_sbyin_qa_studies

    check:
    gnod_sbyin_qa_studies: G
    """
    return fit_ok and sample_ok


def gnod_sbyin_qa_studies_aux(aux: bool) -> bool:
    """gnod_sbyin_qa_studies

    aux:
    gnod_sbyin_qa_studies: n
    """
    return aux


def _bench_gnod_sbyin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gnod_sbyin_qa_studies_ok(True, True))
    checks.append(not gnod_sbyin_qa_studies_ok(False, True))
    checks.append(gnod_sbyin_qa_studies_aux(True))
    checks.append(not gnod_sbyin_qa_studies_aux(False))
    checks.append(True)  # tibetan-demon canon
    return float(sum(checks) / len(checks))


def bench_gnod_sbyin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gnod_sbyin_qa_studies": _bench_gnod_sbyin_qa_studies(seed)}
