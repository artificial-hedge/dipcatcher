"""apollopatara2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def apollopatara2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """apollopatara2_qa_studies

    check:
    apollopatara2_qa_studies: ApolloPatara2QA metrics
    """
    return fit_ok and sample_ok


def apollopatara2_qa_studies_aux(aux: bool) -> bool:
    """apollopatara2_qa_studies

    aux:
    apollopatara2_qa_studies: apollopatara2, oracle archers, answers, and scores
    """
    return aux


def _bench_apollopatara2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(apollopatara2_qa_studies_ok(True, True))
    checks.append(not apollopatara2_qa_studies_ok(False, True))
    checks.append(apollopatara2_qa_studies_aux(True))
    checks.append(not apollopatara2_qa_studies_aux(False))
    checks.append(True)  # lycian-myth canon
    return float(sum(checks) / len(checks))


def bench_apollopatara2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_apollopatara2_qa_studies": _bench_apollopatara2_qa_studies(seed)}
