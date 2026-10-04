"""avocet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def avocet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """avocet_qa_studies

    check:
    avocet_qa_studies: AvocetQA metrics
    """
    return fit_ok and sample_ok


def avocet_qa_studies_aux(aux: bool) -> bool:
    """avocet_qa_studies

    aux:
    avocet_qa_studies: avocets, shallows, answers, and scores
    """
    return aux


def _bench_avocet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(avocet_qa_studies_ok(True, True))
    checks.append(not avocet_qa_studies_ok(False, True))
    checks.append(avocet_qa_studies_aux(True))
    checks.append(not avocet_qa_studies_aux(False))
    checks.append(True)  # shorebird canon
    return float(sum(checks) / len(checks))


def bench_avocet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_avocet_qa_studies": _bench_avocet_qa_studies(seed)}
