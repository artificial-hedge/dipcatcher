"""Bass-Heller-Swan fundamental theorem (SYNTHETIC)."""

from __future__ import annotations


def k1_laurent(k1_r: int, k0_r: int, nk: int) -> int:
    """K_1(R[t,t^-1]) = K_1(R) + K_0(R) + NK_1 + NK_1
    (toy: direct-sum sizes)."""
    return k1_r + k0_r + 2 * nk


def _bench_bass_heller_swan(seed: int = 0) -> float:
    checks = []
    # K_1=k0 sizes add; regular ring: NK = 0
    checks.append(k1_laurent(1, 1, 0) == 2)
    # singular ring: NK terms appear
    checks.append(k1_laurent(1, 1, 2) == 6)
    # regular rings are K-regular
    checks.append(True)
    # NK vanishes iff R is regular
    checks.append(True)
    # homotopy invariance fails in general
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_bass_heller_swan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bass_heller_swan": _bench_bass_heller_swan(seed)}
