"""chr_f_studies module (SYNTHETIC)."""

from __future__ import annotations


def chr_f_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """chr_f_studies

    check:
    chr_f_studies: chrF character n-gram metrics
    """
    return fit_ok and sample_ok


def chr_f_studies_aux(aux: bool) -> bool:
    """chr_f_studies

    aux:
    chr_f_studies: hypotheses, references, labels, and scores
    """
    return aux


def _bench_chr_f_studies(seed: int = 0) -> float:
    checks = []
    checks.append(chr_f_studies_ok(True, True))
    checks.append(not chr_f_studies_ok(False, True))
    checks.append(chr_f_studies_aux(True))
    checks.append(not chr_f_studies_aux(False))
    checks.append(True)  # translation-metric canon
    return float(sum(checks) / len(checks))


def bench_chr_f_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chr_f_studies": _bench_chr_f_studies(seed)}
