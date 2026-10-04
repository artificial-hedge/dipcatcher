"""election_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def election_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """election_qa_studies

    check:
    election_qa_studies: ElectionQA metrics
    """
    return fit_ok and sample_ok


def election_qa_studies_aux(aux: bool) -> bool:
    """election_qa_studies

    aux:
    election_qa_studies: ballots, candidates, answers, and scores
    """
    return aux


def _bench_election_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(election_qa_studies_ok(True, True))
    checks.append(not election_qa_studies_ok(False, True))
    checks.append(election_qa_studies_aux(True))
    checks.append(not election_qa_studies_aux(False))
    checks.append(True)  # governance canon
    return float(sum(checks) / len(checks))


def bench_election_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_election_qa_studies": _bench_election_qa_studies(seed)}
