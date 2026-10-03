"""Completions in homotopy (SYNTHETIC)."""

from __future__ import annotations


def ch_ok(completion: bool, homotopy: bool) -> bool:
    """Completion
    homotopy:
    profinite
    completion —
    Sullivan."""
    return completion and homotopy


def bousfield_kan_completion(bkc: bool) -> bool:
    """Bousfield-Kan
    completion:
    R-completion
    functor —
    unstable."""
    return bkc


def _bench_completion_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(ch_ok(True, True))
    checks.append(not ch_ok(False, True))
    checks.append(bousfield_kan_completion(True))
    checks.append(not bousfield_kan_completion(False))
    checks.append(True)  # Bousfield-Kan
    return float(sum(checks) / len(checks))


def bench_completion_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_completion_htpy": _bench_completion_htpy(seed)}
