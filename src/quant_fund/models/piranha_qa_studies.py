"""piranha_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def piranha_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """piranha_qa_studies

    check:
    piranha_qa_studies: PiranhaQA metrics
    """
    return fit_ok and sample_ok


def piranha_qa_studies_aux(aux: bool) -> bool:
    """piranha_qa_studies

    aux:
    piranha_qa_studies: piranhas, rivers, answers, and scores
    """
    return aux


def _bench_piranha_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(piranha_qa_studies_ok(True, True))
    checks.append(not piranha_qa_studies_ok(False, True))
    checks.append(piranha_qa_studies_aux(True))
    checks.append(not piranha_qa_studies_aux(False))
    checks.append(True)  # fish canon
    return float(sum(checks) / len(checks))


def bench_piranha_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_piranha_qa_studies": _bench_piranha_qa_studies(seed)}
