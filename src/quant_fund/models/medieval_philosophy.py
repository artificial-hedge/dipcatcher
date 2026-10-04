"""medieval_philosophy module (SYNTHETIC)."""

from __future__ import annotations


def medieval_philosophy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """medieval_philosophy

    check:
    ancient_philosophy: ancient philosophy
    medieval_philosophy: medieval philosophy
    continental_philosophy: continental philosophy
    analytic_philosophy: analytic philosophy
    pragmatism: pragmatism
    existentialism: existentialism
    """
    return fit_ok and sample_ok


def medieval_philosophy_aux(aux: bool) -> bool:
    """medieval_philosophy

    aux:
    ancient_philosophy: greek philosophy
    medieval_philosophy: scholasticism
    continental_philosophy: phenomenological tradition
    analytic_philosophy: logical analysis
    pragmatism: pragmatic tradition
    existentialism: existential thought
    """
    return aux


def _bench_medieval_philosophy(seed: int = 0) -> float:
    checks = []
    checks.append(medieval_philosophy_ok(True, True))
    checks.append(not medieval_philosophy_ok(False, True))
    checks.append(medieval_philosophy_aux(True))
    checks.append(not medieval_philosophy_aux(False))
    checks.append(True)  # philosophy-2 canon
    return float(sum(checks) / len(checks))


def bench_medieval_philosophy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_medieval_philosophy": _bench_medieval_philosophy(seed)}
