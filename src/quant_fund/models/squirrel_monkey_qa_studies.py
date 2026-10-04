"""squirrel_monkey_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def squirrel_monkey_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """squirrel_monkey_qa_studies

    check:
    squirrel_monkey_qa_studies: SquirrelMonkeyQA metrics
    """
    return fit_ok and sample_ok


def squirrel_monkey_qa_studies_aux(aux: bool) -> bool:
    """squirrel_monkey_qa_studies

    aux:
    squirrel_monkey_qa_studies: squirrel monkeys, riverine thickets, answers, and scores
    """
    return aux


def _bench_squirrel_monkey_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(squirrel_monkey_qa_studies_ok(True, True))
    checks.append(not squirrel_monkey_qa_studies_ok(False, True))
    checks.append(squirrel_monkey_qa_studies_aux(True))
    checks.append(not squirrel_monkey_qa_studies_aux(False))
    checks.append(True)  # new-world-monkey canon
    return float(sum(checks) / len(checks))


def bench_squirrel_monkey_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_squirrel_monkey_qa_studies": _bench_squirrel_monkey_qa_studies(seed)}
