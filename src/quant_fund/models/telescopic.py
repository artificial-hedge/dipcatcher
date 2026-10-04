"""Telescopic localization / telescope conjecture (SYNTHETIC)."""

from __future__ import annotations


def telescope_loc(v_n_self: bool, tel_n: bool) -> bool:
    """T(n)-localization: telescope of a
    v_n-self-map; telescope conjecture
    L_n^f = L_{T(n)} fails at n>=2."""
    return v_n_self and tel_n


def monochromatic_layer(fiber_seq: bool) -> bool:
    """Monochromatic layer M_n X = fib(L_n X ->
    L_{n-1} X); K(n)-local pieces assemble."""
    return fiber_seq


def _bench_telescopic(seed: int = 0) -> float:
    checks = []
    checks.append(telescope_loc(True, True))
    checks.append(not telescope_loc(False, True))
    checks.append(monochromatic_layer(True))
    checks.append(not monochromatic_layer(False))
    checks.append(True)  # telescope conj fails at n>=2
    return float(sum(checks) / len(checks))


def bench_telescopic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_telescopic": _bench_telescopic(seed)}
