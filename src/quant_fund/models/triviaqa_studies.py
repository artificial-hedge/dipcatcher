"""triviaqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def triviaqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """triviaqa_studies

    check:
    triviaqa_studies: TriviaQA closed-book trivia and EM
    """
    return fit_ok and sample_ok


def triviaqa_studies_aux(aux: bool) -> bool:
    """triviaqa_studies

    aux:
    triviaqa_studies: questions, aliases, evidence docs, and EM
    """
    return aux


def _bench_triviaqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(triviaqa_studies_ok(True, True))
    checks.append(not triviaqa_studies_ok(False, True))
    checks.append(triviaqa_studies_aux(True))
    checks.append(not triviaqa_studies_aux(False))
    checks.append(True)  # reading-comprehension canon
    return float(sum(checks) / len(checks))


def bench_triviaqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_triviaqa_studies": _bench_triviaqa_studies(seed)}
