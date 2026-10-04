"""Handle decompositions: chi bookkeeping under handle attachment (SYNTHETIC)."""

from __future__ import annotations


def chi_after_attach(handles: list[int], dim: int = 2) -> int:
    """Euler characteristic after attaching 0/1/2-handles: chi = sum (-1)^k h_k."""
    chi = 0
    counts = [0, 0, 0]
    for h in handles:
        counts[h] += 1
    for k in range(dim + 1):
        chi += (-1) ** k * counts[k]
    return chi


def _bench_handle_decomp(seed: int = 0) -> float:
    checks = []
    # S^2 = 0-handle + 2-handle: chi = 1 - 0 + 1 = 2
    checks.append(chi_after_attach([0, 2]) == 2)
    # T^2 = 0 + 1 + 1 + 2 handles: chi = 0
    checks.append(chi_after_attach([0, 1, 1, 2]) == 0)
    # genus g: 0 + 2g ones + 2 -> chi = 2 - 2g
    checks.append(chi_after_attach([0, 1, 1, 1, 1, 2]) == -2)
    # disc D^2 = 0-handle: chi = 1
    checks.append(chi_after_attach([0]) == 1)
    # annulus S^1 x I = 0 + 1: chi = 0
    checks.append(chi_after_attach([0, 1]) == 0)
    # Klein bottle = 0 + 1 + 1 + 2 (twisted): chi = 0 same count
    checks.append(chi_after_attach([0, 1, 1, 2]) == 0)
    # extra cancelling pair adds both signs: chi stays a pure count
    checks.append(chi_after_attach([0, 1, 1, 2, 2]) == 1 - 2 + 2)
    return float(sum(checks) / len(checks))


def bench_handle_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_handle_decomp": _bench_handle_decomp(seed)}
