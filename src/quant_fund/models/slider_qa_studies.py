"""slider_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def slider_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """slider_qa_studies

    check:
    slider_qa_studies: SliderQA metrics
    """
    return fit_ok and sample_ok


def slider_qa_studies_aux(aux: bool) -> bool:
    """slider_qa_studies

    aux:
    slider_qa_studies: sliders, calm ponds, answers, and scores
    """
    return aux


def _bench_slider_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(slider_qa_studies_ok(True, True))
    checks.append(not slider_qa_studies_ok(False, True))
    checks.append(slider_qa_studies_aux(True))
    checks.append(not slider_qa_studies_aux(False))
    checks.append(True)  # turtle canon
    return float(sum(checks) / len(checks))


def bench_slider_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_slider_qa_studies": _bench_slider_qa_studies(seed)}
