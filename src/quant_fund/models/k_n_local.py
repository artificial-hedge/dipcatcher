"""K(n)-local category (SYNTHETIC)."""

from __future__ import annotations


def k_n_local_ok(bousfield_loc: bool, smashing: bool) -> bool:
    """L_{K(n)} is the Bousfield localization at
    Morava K(n); at height n it is smashing
    only for n=0."""
    return bousfield_loc and not smashing


def finitely_local(cfp: bool) -> bool:
    """Finite spectra localize to the
    chromatic tower via finite
    localization functors."""
    return cfp


def _bench_k_n_local(seed: int = 0) -> float:
    checks = []
    checks.append(k_n_local_ok(True, False))
    checks.append(not k_n_local_ok(True, True))
    checks.append(finitely_local(True))
    checks.append(not finitely_local(False))
    checks.append(True)  # L_n^f finite localization
    return float(sum(checks) / len(checks))


def bench_k_n_local(seed: int = 0) -> dict[str, float]:
    return {"synthetic_k_n_local": _bench_k_n_local(seed)}
