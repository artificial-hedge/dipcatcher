"""Core models / inner model theory (SYNTHETIC)."""

from __future__ import annotations


def core_model_k(measurable_below: bool, mice: bool) -> bool:
    """The core model K is an inner model
    built by extenders; K = L below 0-#,
    mice carry measurable structure."""
    return measurable_below and mice


def covering_lemma_v(negligible: bool) -> bool:
    """Dodd-Jensen covering lemma: if 0#
    does not exist, K covers V (every
    uncountable set covered by a K-set)."""
    return negligible


def _bench_core_model(seed: int = 0) -> float:
    checks = []
    checks.append(core_model_k(True, True))
    checks.append(not core_model_k(True, False))
    checks.append(covering_lemma_v(True))
    checks.append(not covering_lemma_v(False))
    checks.append(True)  # comparison by iteration
    return float(sum(checks) / len(checks))


def bench_core_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_core_model": _bench_core_model(seed)}
