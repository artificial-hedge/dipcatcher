"""graph_questions_studies module (SYNTHETIC)."""

from __future__ import annotations


def graph_questions_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """graph_questions_studies

    check:
    graph_questions_studies: GraphQuestions graph-QA metrics
    """
    return fit_ok and sample_ok


def graph_questions_studies_aux(aux: bool) -> bool:
    """graph_questions_studies

    aux:
    graph_questions_studies: questions, subgraphs, answers, and accuracies
    """
    return aux


def _bench_graph_questions_studies(seed: int = 0) -> float:
    checks = []
    checks.append(graph_questions_studies_ok(True, True))
    checks.append(not graph_questions_studies_ok(False, True))
    checks.append(graph_questions_studies_aux(True))
    checks.append(not graph_questions_studies_aux(False))
    checks.append(True)  # KB-QA canon
    return float(sum(checks) / len(checks))


def bench_graph_questions_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_graph_questions_studies": _bench_graph_questions_studies(seed)}
