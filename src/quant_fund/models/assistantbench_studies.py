"""assistantbench_studies module (SYNTHETIC)."""

from __future__ import annotations


def assistantbench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """assistantbench_studies

    check:
    assistantbench_studies: AssistantBench web-task metrics
    """
    return fit_ok and sample_ok


def assistantbench_studies_aux(aux: bool) -> bool:
    """assistantbench_studies

    aux:
    assistantbench_studies: tasks, answers, time, and accuracy scores
    """
    return aux


def _bench_assistantbench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(assistantbench_studies_ok(True, True))
    checks.append(not assistantbench_studies_ok(False, True))
    checks.append(assistantbench_studies_aux(True))
    checks.append(not assistantbench_studies_aux(False))
    checks.append(True)  # agentic-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_assistantbench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_assistantbench_studies": _bench_assistantbench_studies(seed)}
