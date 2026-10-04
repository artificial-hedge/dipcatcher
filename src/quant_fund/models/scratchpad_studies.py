"""scratchpad_studies module (SYNTHETIC)."""

from __future__ import annotations


def scratchpad_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """scratchpad_studies

    check:
    scratchpad_studies: intermediate-state arithmetic and tooling/buffers and ops
    """
    return fit_ok and sample_ok


def scratchpad_studies_aux(aux: bool) -> bool:
    """scratchpad_studies

    aux:
    scratchpad_studies: scratchpad prediction and length scaling/digits and carries
    """
    return aux


def _bench_scratchpad_studies(seed: int = 0) -> float:
    checks = []
    checks.append(scratchpad_studies_ok(True, True))
    checks.append(not scratchpad_studies_ok(False, True))
    checks.append(scratchpad_studies_aux(True))
    checks.append(not scratchpad_studies_aux(False))
    checks.append(True)  # reasoning/CoT canon
    return float(sum(checks) / len(checks))


def bench_scratchpad_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scratchpad_studies": _bench_scratchpad_studies(seed)}
