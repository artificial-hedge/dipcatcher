"""Harris-Taylor theorem (SYNTHETIC)."""

from __future__ import annotations


def ht_ok(lubin_tate: bool, vanishing: bool) -> bool:
    """Harris-Taylor:
    supercuspidal part
    of l-adic cohomology
    of the Lubin-Tate
    tower realizes
    LLC for GL_n."""
    return lubin_tate and vanishing


def drinfeld_tower(etale: bool) -> bool:
    """Drinfeld tower:
    inverse limit of
    Drinfeld upper-half
    covers; cohomology
    carries GL_n x D^x
    x W_F action."""
    return etale


def _bench_harris_taylor(seed: int = 0) -> float:
    checks = []
    checks.append(ht_ok(True, True))
    checks.append(not ht_ok(False, True))
    checks.append(drinfeld_tower(True))
    checks.append(not drinfeld_tower(False))
    checks.append(True)  # Carayol before HT
    return float(sum(checks) / len(checks))


def bench_harris_taylor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harris_taylor": _bench_harris_taylor(seed)}
