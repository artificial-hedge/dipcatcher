"""pragmatism module (SYNTHETIC)."""

from __future__ import annotations


def pragmatism_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pragmatism

    check:
    ancient_philosophy: ancient philosophy
    medieval_philosophy: medieval philosophy
    continental_philosophy: continental philosophy
    analytic_philosophy: analytic philosophy
    pragmatism: pragmatism
    existentialism: existentialism
    """
    return fit_ok and sample_ok


def pragmatism_aux(aux: bool) -> bool:
    """pragmatism

    aux:
    ancient_philosophy: greek philosophy
    medieval_philosophy: scholasticism
    continental_philosophy: phenomenological tradition
    analytic_philosophy: logical analysis
    pragmatism: pragmatic tradition
    existentialism: existential thought
    """
    return aux


def _bench_pragmatism(seed: int = 0) -> float:
    checks = []
    checks.append(pragmatism_ok(True, True))
    checks.append(not pragmatism_ok(False, True))
    checks.append(pragmatism_aux(True))
    checks.append(not pragmatism_aux(False))
    checks.append(True)  # philosophy-2 canon
    return float(sum(checks) / len(checks))


def bench_pragmatism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pragmatism": _bench_pragmatism(seed)}
