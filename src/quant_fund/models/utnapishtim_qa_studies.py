"""utnapishtim_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def utnapishtim_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """utnapishtim_qa_studies

    check:
    utnapishtim_qa_studies: UtnapishtimQA metrics
    """
    return fit_ok and sample_ok


def utnapishtim_qa_studies_aux(aux: bool) -> bool:
    """utnapishtim_qa_studies

    aux:
    utnapishtim_qa_studies: utnapishtim, flood farers, answers, and scores
    """
    return aux


def _bench_utnapishtim_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(utnapishtim_qa_studies_ok(True, True))
    checks.append(not utnapishtim_qa_studies_ok(False, True))
    checks.append(utnapishtim_qa_studies_aux(True))
    checks.append(not utnapishtim_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-2 canon
    return float(sum(checks) / len(checks))


def bench_utnapishtim_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_utnapishtim_qa_studies": _bench_utnapishtim_qa_studies(seed)}
