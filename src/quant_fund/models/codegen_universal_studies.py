"""codegen_universal_studies module (SYNTHETIC)."""

from __future__ import annotations


def codegen_universal_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """codegen_universal_studies

    check:
    codegen_universal_studies: Universal code-generation metrics
    """
    return fit_ok and sample_ok


def codegen_universal_studies_aux(aux: bool) -> bool:
    """codegen_universal_studies

    aux:
    codegen_universal_studies: specs, programs, tests, and pass rates
    """
    return aux


def _bench_codegen_universal_studies(seed: int = 0) -> float:
    checks = []
    checks.append(codegen_universal_studies_ok(True, True))
    checks.append(not codegen_universal_studies_ok(False, True))
    checks.append(codegen_universal_studies_aux(True))
    checks.append(not codegen_universal_studies_aux(False))
    checks.append(True)  # code-eval-4 canon
    return float(sum(checks) / len(checks))


def bench_codegen_universal_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_codegen_universal_studies": _bench_codegen_universal_studies(seed)}
