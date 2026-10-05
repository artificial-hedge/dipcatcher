"""egui_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def egui_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """egui_qa_studies

    check:
    egui_qa_studies: E
    """
    return fit_ok and sample_ok


def egui_qa_studies_aux(aux: bool) -> bool:
    """egui_qa_studies

    aux:
    egui_qa_studies: g
    """
    return aux


def _bench_egui_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(egui_qa_studies_ok(True, True))
    checks.append(not egui_qa_studies_ok(False, True))
    checks.append(egui_qa_studies_aux(True))
    checks.append(not egui_qa_studies_aux(False))
    checks.append(True)  # chinese-underworld canon
    return float(sum(checks) / len(checks))


def bench_egui_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_egui_qa_studies": _bench_egui_qa_studies(seed)}
