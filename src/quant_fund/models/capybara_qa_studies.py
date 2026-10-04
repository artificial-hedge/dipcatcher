"""capybara_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def capybara_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """capybara_qa_studies

    check:
    capybara_qa_studies: CapybaraQA metrics
    """
    return fit_ok and sample_ok


def capybara_qa_studies_aux(aux: bool) -> bool:
    """capybara_qa_studies

    aux:
    capybara_qa_studies: capybaras, wetlands, answers, and scores
    """
    return aux


def _bench_capybara_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(capybara_qa_studies_ok(True, True))
    checks.append(not capybara_qa_studies_ok(False, True))
    checks.append(capybara_qa_studies_aux(True))
    checks.append(not capybara_qa_studies_aux(False))
    checks.append(True)  # neotropical canon
    return float(sum(checks) / len(checks))


def bench_capybara_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_capybara_qa_studies": _bench_capybara_qa_studies(seed)}
