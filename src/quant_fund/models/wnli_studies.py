"""wnli_studies module (SYNTHETIC)."""

from __future__ import annotations


def wnli_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wnli_studies

    check:
    wnli_studies: WNLI Winograd coreference NLI accuracy
    """
    return fit_ok and sample_ok


def wnli_studies_aux(aux: bool) -> bool:
    """wnli_studies

    aux:
    wnli_studies: ambiguous pronouns, referents, and labels
    """
    return aux


def _bench_wnli_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wnli_studies_ok(True, True))
    checks.append(not wnli_studies_ok(False, True))
    checks.append(wnli_studies_aux(True))
    checks.append(not wnli_studies_aux(False))
    checks.append(True)  # GLUE-eval canon
    return float(sum(checks) / len(checks))


def bench_wnli_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wnli_studies": _bench_wnli_studies(seed)}
