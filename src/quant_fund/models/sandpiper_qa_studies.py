"""sandpiper_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sandpiper_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sandpiper_qa_studies

    check:
    sandpiper_qa_studies: SandpiperQA metrics
    """
    return fit_ok and sample_ok


def sandpiper_qa_studies_aux(aux: bool) -> bool:
    """sandpiper_qa_studies

    aux:
    sandpiper_qa_studies: sandpipers, mudflats, answers, and scores
    """
    return aux


def _bench_sandpiper_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sandpiper_qa_studies_ok(True, True))
    checks.append(not sandpiper_qa_studies_ok(False, True))
    checks.append(sandpiper_qa_studies_aux(True))
    checks.append(not sandpiper_qa_studies_aux(False))
    checks.append(True)  # shorebird canon
    return float(sum(checks) / len(checks))


def bench_sandpiper_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sandpiper_qa_studies": _bench_sandpiper_qa_studies(seed)}
